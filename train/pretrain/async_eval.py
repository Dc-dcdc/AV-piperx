"""基于独立进程和不可变模型快照的异步预训练评估。"""

from __future__ import annotations

import atexit
import json
import logging
import multiprocessing as mp
import queue
import re
import shutil
import traceback
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


_EVAL_METRIC_SUFFIX_PATTERN = re.compile(
    r"_sr=[+-]?(?:\d+(?:\.\d*)?|\.\d+|inf|nan)"
    r"_ar=[+-]?(?:\d+(?:\.\d*)?|\.\d+|inf|nan)$",
    flags=re.IGNORECASE,
)


def make_evaluated_checkpoint_identifier(
    base_identifier: str,
    *,
    success_rate: float,
    average_reward: float,
) -> str:
    """在稳定的 step/loss 名称后追加同步评估使用的 sr/ar 后缀。"""

    base_identifier = _EVAL_METRIC_SUFFIX_PATTERN.sub("", str(base_identifier))
    return (
        f"{base_identifier}_sr={float(success_rate) * 100.0:.1f}"
        f"_ar={float(average_reward):.2f}"
    )


def rename_evaluated_checkpoint(
    checkpoint_path: str | Path,
    *,
    success_rate: float,
    average_reward: float,
) -> Path:
    """异步指标返回后重命名 checkpoint，并只在需要时修复 ``last`` 软链接。"""

    checkpoint_path = Path(checkpoint_path).absolute()
    renamed_path = checkpoint_path.with_name(
        make_evaluated_checkpoint_identifier(
            checkpoint_path.name,
            success_rate=success_rate,
            average_reward=average_reward,
        )
    )
    if renamed_path == checkpoint_path:
        return checkpoint_path
    if renamed_path.exists():
        raise FileExistsError(f"评估后 checkpoint 名称已存在，拒绝覆盖: {renamed_path}")

    last_link = checkpoint_path.parent / "last"
    last_points_to_checkpoint = (
        last_link.is_symlink()
        and last_link.resolve(strict=False) == checkpoint_path.resolve(strict=False)
    )

    checkpoint_path.rename(renamed_path)
    if last_points_to_checkpoint:
        last_link.unlink()
        last_link.symlink_to(renamed_path, target_is_directory=True)

    logging.info("checkpoint 已按评估指标重命名: %s", renamed_path)
    return renamed_path


def migrate_evaluated_checkpoint_names(
    checkpoints_dir: str | Path,
) -> dict[str, str]:
    """根据已有 ``eval_metrics.json`` 补齐历史 checkpoint 的 sr/ar 名称。"""

    checkpoints_dir = Path(checkpoints_dir).absolute()
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"checkpoint 目录不存在: {checkpoints_dir}")

    renamed_paths: dict[str, str] = {}
    checkpoint_dirs = sorted(
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.split("_", 1)[0].isdigit()
    )
    for checkpoint_path in checkpoint_dirs:
        metrics_path = checkpoint_path / "eval_metrics.json"
        if not metrics_path.is_file():
            continue
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        aggregated = metrics.get("aggregated", {})
        if "success_rate" not in aggregated or "average_reward" not in aggregated:
            logging.warning("跳过缺少成功率/奖励的历史 checkpoint: %s", checkpoint_path)
            continue

        old_path = checkpoint_path.absolute()
        new_path = rename_evaluated_checkpoint(
            old_path,
            success_rate=float(aggregated["success_rate"]),
            average_reward=float(aggregated["average_reward"]),
        )
        if new_path != old_path:
            renamed_paths[str(old_path)] = str(new_path)

    records_path = checkpoints_dir / "top_k_records.json"
    if renamed_paths and records_path.is_file():
        with records_path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        latest = records.get("latest")
        if latest:
            records["latest"] = renamed_paths.get(
                str(Path(latest).absolute()),
                latest,
            )
        for item in records.get("top_k", []):
            item_path = item.get("path")
            if item_path:
                item["path"] = renamed_paths.get(
                    str(Path(item_path).absolute()),
                    item_path,
                )

        temporary_records_path = records_path.with_suffix(".json.tmp")
        with temporary_records_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=4, ensure_ascii=False)
        temporary_records_path.replace(records_path)

    return renamed_paths


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"异步评估不支持amp_dtype={dtype_name!r}")


def _async_eval_worker(
    request_queue,
    result_queue,
    *,
    policy_name: str,
    env_id: str,
    obs_cameras: list[str],
    eval_cfg: dict[str, Any],
    eval_device: str,
    use_amp: bool,
    amp_dtype: str,
):
    """子进程入口；CUDA模型、MuJoCo环境和随机状态均与训练进程隔离。"""
    eval_env = None
    try:
        # 延迟导入，确保spawn子进程先建立自己的CUDA/MuJoCo运行时。
        from lerobot.common.policies.factory import get_policy_and_config_classes
        from train.pretrain.eval_train import custom_eval_policy, make_eval_env

        device = torch.device(eval_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"异步评估请求使用{eval_device}，但当前进程无法访问CUDA。"
                )
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"异步评估设备{eval_device}不存在；"
                    f"当前可见GPU数量为{torch.cuda.device_count()}。"
                )
            torch.cuda.set_device(device)

        cfg_eval = OmegaConf.create(eval_cfg)
        eval_env = make_eval_env(env_id, obs_cameras, cfg_eval)
        policy_cls, _ = get_policy_and_config_classes(policy_name)
        autocast_dtype = _resolve_amp_dtype(amp_dtype)
        torch.set_grad_enabled(False)
        result_queue.put(
            {
                "kind": "worker_ready",
                "device": str(device),
            }
        )
    except Exception:
        result_queue.put(
            {
                "kind": "worker_error",
                "error": traceback.format_exc(),
            }
        )
        if eval_env is not None and hasattr(eval_env, "close"):
            eval_env.close()
        return

    while True:
        request = request_queue.get()
        if request is None:
            break

        policy = None
        try:
            policy = policy_cls.from_pretrained(
                request["snapshot_path"],
                strict=True,
            )
            policy.to(device)
            policy.eval()
            policy.reset()

            autocast_context = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                eval_info = custom_eval_policy(
                    env=eval_env,
                    policy=policy,
                    cfg_eval=cfg_eval,
                    videos_dir=request["videos_dir"],
                    device=device,
                )
            result_queue.put(
                {
                    "kind": "eval_result",
                    "ok": True,
                    "request": request,
                    "eval_info": eval_info,
                }
            )
        except Exception:
            result_queue.put(
                {
                    "kind": "eval_result",
                    "ok": False,
                    "request": request,
                    "error": traceback.format_exc(),
                }
            )
        finally:
            del policy
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if eval_env is not None and hasattr(eval_env, "close"):
        eval_env.close()


class AsyncEvalController:
    """训练主进程中的异步评估队列和子进程生命周期管理器。"""

    def __init__(
        self,
        *,
        policy_name: str,
        env_id: str,
        obs_cameras: list[str],
        eval_cfg,
        eval_device: str,
        use_amp: bool,
        amp_dtype: str,
        out_dir: str | Path,
        max_pending: int = 1,
        startup_timeout_s: float = 180.0,
        shutdown_timeout_s: float = 30.0,
    ):
        if max_pending <= 0:
            raise ValueError("eval.max_pending必须大于0。")
        self.max_pending = int(max_pending)
        self.startup_timeout_s = float(startup_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.snapshot_root = Path(out_dir) / "eval" / "async_snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

        context = mp.get_context("spawn")
        self._request_queue = context.Queue(maxsize=self.max_pending)
        self._result_queue = context.Queue()
        self._pending: dict[str, dict[str, Any]] = {}
        self._atexit_registered = False
        self._process = context.Process(
            target=_async_eval_worker,
            kwargs={
                "request_queue": self._request_queue,
                "result_queue": self._result_queue,
                "policy_name": str(policy_name),
                "env_id": str(env_id),
                "obs_cameras": list(obs_cameras),
                "eval_cfg": OmegaConf.to_container(eval_cfg, resolve=True),
                "eval_device": str(eval_device),
                "use_amp": bool(use_amp),
                "amp_dtype": str(amp_dtype),
            },
            name="async-pretrain-evaluator",
            daemon=False,
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def has_capacity(self) -> bool:
        return self.pending_count < self.max_pending

    def start(self):
        self._process.start()
        atexit.register(self._force_close_at_exit)
        self._atexit_registered = True
        try:
            message = self._result_queue.get(timeout=self.startup_timeout_s)
        except queue.Empty as exc:
            self.close(force=True)
            raise RuntimeError(
                f"异步评估进程在{self.startup_timeout_s:g}秒内未完成初始化。"
            ) from exc

        if message.get("kind") != "worker_ready":
            self.close(force=True)
            raise RuntimeError(
                "异步评估进程初始化失败：\n"
                f"{message.get('error', message)}"
            )
        logging.info(
            "异步评估进程已启动: pid=%s, device=%s, max_pending=%d",
            self._process.pid,
            message["device"],
            self.max_pending,
        )

    def _force_close_at_exit(self):
        """训练异常或Ctrl-C退出时终止非daemon评估进程，避免解释器等待挂起。"""
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)

    def save_temporary_snapshot(
        self,
        policy,
        *,
        step: int,
    ) -> tuple[Path, Path]:
        """同步保存一次不可变权重；返回模型目录和完成后可删除的任务目录。"""
        task_dir = self.snapshot_root / f"{step:09d}_{uuid.uuid4().hex[:10]}"
        model_dir = task_dir / "pretrained_model"
        policy.save_pretrained(model_dir)
        return model_dir, task_dir

    def submit(
        self,
        *,
        step: int,
        train_loss: float,
        base_identifier: str,
        snapshot_path: str | Path,
        videos_dir: str | Path,
        checkpoint_path: str | Path | None,
        cleanup_snapshot_dir: str | Path | None,
    ) -> bool:
        if not self.has_capacity:
            return False
        if not self._process.is_alive():
            raise RuntimeError("异步评估进程已退出，无法提交新的评估任务。")

        task_id = f"{step}-{uuid.uuid4().hex}"
        request = {
            "task_id": task_id,
            "step": int(step),
            "train_loss": float(train_loss),
            "base_identifier": str(base_identifier),
            "snapshot_path": str(Path(snapshot_path).absolute()),
            "videos_dir": str(Path(videos_dir).absolute()),
            "checkpoint_path": (
                str(Path(checkpoint_path).absolute())
                if checkpoint_path is not None
                else None
            ),
            "cleanup_snapshot_dir": (
                str(Path(cleanup_snapshot_dir).absolute())
                if cleanup_snapshot_dir is not None
                else None
            ),
        }
        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            return False
        self._pending[task_id] = request
        return True

    def poll(self) -> list[dict[str, Any]]:
        results = []
        while True:
            try:
                message = self._result_queue.get_nowait()
            except queue.Empty:
                break

            kind = message.get("kind")
            if kind == "worker_error":
                raise RuntimeError(
                    "异步评估进程异常退出：\n"
                    f"{message.get('error', message)}"
                )
            if kind != "eval_result":
                logging.warning("忽略未知异步评估消息: %s", message)
                continue

            task_id = message["request"]["task_id"]
            self._pending.pop(task_id, None)
            results.append(message)

        if self._pending and not self._process.is_alive():
            raise RuntimeError(
                "异步评估进程已退出，但仍有"
                f"{len(self._pending)}个任务没有返回结果。"
            )
        return results

    def cleanup_result_snapshot(self, result: dict[str, Any]):
        cleanup_path_value = result["request"].get("cleanup_snapshot_dir")
        if not cleanup_path_value:
            return
        cleanup_path = Path(cleanup_path_value).absolute()
        snapshot_root = self.snapshot_root.absolute()
        if not cleanup_path.is_relative_to(snapshot_root):
            raise RuntimeError(
                f"拒绝删除异步评估快照目录之外的路径: {cleanup_path}"
            )
        shutil.rmtree(cleanup_path, ignore_errors=True)

    def close(self, *, force: bool = False):
        if self._atexit_registered:
            atexit.unregister(self._force_close_at_exit)
            self._atexit_registered = False
        if not self._process.is_alive():
            return
        if self._pending and not force:
            raise RuntimeError(
                f"仍有{len(self._pending)}个异步评估任务，不能直接关闭评估进程。"
            )
        if force:
            self._process.terminate()
        else:
            self._request_queue.put(None)
        self._process.join(timeout=self.shutdown_timeout_s)
        if self._process.is_alive():
            logging.warning("异步评估进程未按时退出，将强制终止。")
            self._process.terminate()
            self._process.join(timeout=5.0)


def finalize_async_eval_result(
    result: dict[str, Any],
    *,
    logger,
    cfg,
    manager,
    logging_step: int,
) -> dict[str, float | int] | None:
    """在训练主进程中记录异步指标、归档视频并更新Top-K。"""
    from train.pretrain.eval_train import build_eval_log_metrics

    request = result["request"]
    evaluated_step = int(request["step"])
    train_loss = float(request["train_loss"])
    checkpoint_path = (
        Path(request["checkpoint_path"])
        if request.get("checkpoint_path")
        else None
    )

    if not result.get("ok", False):
        logging.error(
            "Step %d异步评估失败，训练继续运行：\n%s",
            evaluated_step,
            result.get("error", "未知错误"),
        )
        if checkpoint_path is not None and manager is not None:
            manager.release(checkpoint_path)
            if checkpoint_path.exists():
                manager.update(evaluated_step, train_loss, checkpoint_path)
        return None

    eval_info = result["eval_info"]
    metrics = build_eval_log_metrics(eval_info)
    metrics["checkpoint_step"] = evaluated_step
    metrics["evaluation_lag_steps"] = max(0, int(logging_step) - evaluated_step)

    # W&B要求显式step单调递增，因此异步结果记录在当前训练step，同时额外保存
    # checkpoint_step，保证能够追溯它实际评估的模型。
    logger.log_dict(metrics, int(logging_step), mode="eval")
    video_paths = list(eval_info.get("video_paths", []))
    wandb_enabled = bool(getattr(getattr(cfg, "wandb", {}), "enable", False))
    if wandb_enabled and video_paths:
        logger.log_video(video_paths[0], int(logging_step), mode="eval")

    aggregated = eval_info["aggregated"]
    success_rate = float(aggregated["success_rate"])
    average_reward = float(aggregated["average_reward"])
    logging.info(
        "异步评估完成: checkpoint_step=%d, log_step=%d, "
        "success=%.1f%%, reward=%.2f, lag=%d steps",
        evaluated_step,
        int(logging_step),
        success_rate * 100.0,
        average_reward,
        metrics["evaluation_lag_steps"],
    )

    if checkpoint_path is not None and checkpoint_path.exists():
        original_checkpoint_path = checkpoint_path
        checkpoint_path = rename_evaluated_checkpoint(
            checkpoint_path,
            success_rate=success_rate,
            average_reward=average_reward,
        )
        metrics_path = checkpoint_path / "eval_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "checkpoint_step": evaluated_step,
                    "logged_at_step": int(logging_step),
                    "train_loss": train_loss,
                    "aggregated": aggregated,
                    "episodes": eval_info.get("episodes", []),
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        videos_dir = Path(request["videos_dir"])
        if videos_dir.exists():
            target_video_path = checkpoint_path / "eval_videos"
            if target_video_path.exists():
                logging.warning(
                    "checkpoint中已存在eval_videos，保留新的异步视频目录: %s",
                    videos_dir,
                )
            else:
                shutil.move(str(videos_dir), str(target_video_path))
                logging.info("异步评估视频已归档至: %s", target_video_path)

        if manager is not None:
            manager.release(original_checkpoint_path)
            manager.update(
                evaluated_step,
                train_loss,
                checkpoint_path,
                reward=average_reward,
                success_rate=success_rate,
            )

    return metrics
