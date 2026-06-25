import os
DETERMINISTIC_EVAL = os.environ.get("DPPO_EVAL_DETERMINISTIC", "0").lower() in {"1", "true", "yes"}
if DETERMINISTIC_EVAL:
    # 确定性模式会牺牲速度；只在需要逐 bit 复现实验时开启。
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# 独立评估优先使用 EGL 离屏渲染，避免 GLFW/X11 窗口后端带来的额外波动。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
import torch
import csv
import gc
import logging
import numpy as np
import imageio
import json
import re
import shutil
from pathlib import Path
from contextlib import nullcontext
import gymnasium as gym
import yaml
from tqdm import tqdm
import random
import time
import sys
from types import SimpleNamespace
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT_PATH = Path(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import env.task.sim_envs


def seed_runtime(seed: int):
    """重置一次运行时 RNG；评估时每个 episode 都会调用，避免上一局步数污染下一局扩散噪声。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def seed_env_spaces(env, seed: int):
    """Gym space 自己也可能持有 RNG，显式对齐到当前 episode seed。"""
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        if hasattr(space, "seed"):
            space.seed(seed)

def configure_torch_runtime(deterministic: bool):
    if deterministic:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.backends.mkldnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        return

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False)


def prepare_policy_observation(raw_obs: dict, expected_keys: set[str], device) -> dict[str, torch.Tensor]:
    """只转换 policy 真正需要的观测键，避免把录像相机也送进预处理。"""
    batch = {}

    if "observation.state" in expected_keys:
        state = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device, non_blocking=True)

    if "observation.environment_state" in expected_keys and "environment_state" in raw_obs:
        env_state = np.asarray(raw_obs["environment_state"], dtype=np.float32)
        batch["observation.environment_state"] = torch.from_numpy(env_state).unsqueeze(0).to(device, non_blocking=True)

    pixels = raw_obs.get("pixels", {})
    for key in expected_keys:
        if not key.startswith("observation.images."):
            continue
        camera = key.removeprefix("observation.images.")
        if camera not in pixels:
            raise KeyError(f"环境观测中缺少策略需要的相机: {camera}")

        image = pixels[camera]
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image).to(device, non_blocking=True)
        image_tensor = image_tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
        batch[key] = image_tensor.to(dtype=torch.float32).div_(255.0)

    return batch


def patch_act_position_embedding_for_determinism():
    """ACT 的 CUDA cumsum 非确定；评估时用等价 arange 坐标替代。"""
    try:
        from lerobot.common.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d
    except Exception as exc:
        logging.warning(f"无法应用 ACT 确定性补丁: {exc}")
        return

    if getattr(ACTSinusoidalPositionEmbedding2d, "_dppo_deterministic_patch", False):
        return

    def deterministic_forward(self, x):
        height, width = x.shape[-2:]
        y_range = torch.arange(1, height + 1, dtype=torch.float32, device=x.device)
        y_range = y_range.view(1, height, 1).expand(1, height, width)
        x_range = torch.arange(1, width + 1, dtype=torch.float32, device=x.device)
        x_range = x_range.view(1, 1, width).expand(1, height, width)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

    ACTSinusoidalPositionEmbedding2d.forward = deterministic_forward
    ACTSinusoidalPositionEmbedding2d._dppo_deterministic_patch = True
    logging.info("已启用 ACT 评估确定性补丁: positional embedding 使用 arange 替代 CUDA cumsum")

def custom_eval_policy(env, policy, cfg_eval, videos_dir, device):
    """
    完全自主实现的评估代码。没有任何黑盒。
    接收标准 Gym 环境，处理图像归一化，跑策略推理，保存视频。
    """
    policy.eval() # 必须开启评估模式
    successes = []
    rewards = []
    episode_records = []

    # 用来动态记录实际保存的视频路径
    saved_video_paths = []
    
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # 从配置中动态读取参数，提供后备默认值防崩溃
    n_episodes = getattr(cfg_eval, "n_episodes", 10)
    max_rendered = getattr(cfg_eval, "max_episodes_rendered", 4)
    fps = getattr(cfg_eval, "fps", 25)
    max_steps = getattr(cfg_eval, "max_steps", 300)
    raw_camera = getattr(cfg_eval, "render_camera", 'overhead_cam')
    render_cameras = [raw_camera] if isinstance(raw_camera, str) else list(raw_camera)
    expected_keys = set(policy.config.input_shapes)
    # 从配置中提取基础种子，默认为 1000
    base_seed = getattr(cfg_eval, "seed", 1000)
    # 计算所有评估回合推理的总耗时
    global_real_inference_times = []
    # 动作执行循环
    for ep in tqdm(range(n_episodes), leave=False):
        # 计算当前回合的专属种子，并传给环境
        ep_seed = base_seed + ep
        # 先固定全局 RNG，再 reset 环境；这样 reset 内部若用 random/np/torch 也可复现。
        seed_runtime(ep_seed)
        seed_env_spaces(env, ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        done = False
        frames_by_camera = {camera: [] for camera in render_cameras} if ep < max_rendered else {}
        ep_reward = 0
        # LeRobot/DPPO 的 Policy 内置了 action chunking 队列
        policy.reset() # 清空模型的动作缓冲历史
        # policy.reset/env.reset 之后再对齐一次，保证 diffusion select_action 的首个噪声固定。
        seed_runtime(ep_seed)

        # 🌟 1. 每回合新建一个列表，静默记录本回合的所有耗时
        ep_inference_times = []
        steps_taken = 0
        for step in range(max_steps):
            steps_taken = step + 1
            # 1. 如果还在需要渲染的额度内，才调用渲染 (提升非渲染 episode 的评估速度)
            if ep < max_rendered:
                for camera in render_cameras:
                    frames_by_camera[camera].append(env.unwrapped.render([camera])) # gym创建需要加上 .unwrapped

            # 2. 只转换模型真正需要的输入特征，推入 GPU。
            obs = prepare_policy_observation(obs, expected_keys, device)
            # ==========================================
            # ⏱️ 开始计时：使用高精度的 perf_counter
            # ==========================================
            start_time = time.perf_counter()

            # 3. 推理获取动作
            with torch.inference_mode():
                # 使用lerobot自带的推理函数，obs是单帧的，模型会自动处理历史动作的拼接和缓存
                action = policy.select_action(obs) # 这里每次取出一个动作，推理依旧一次生成8个动作，只是一个个往外取
            # 4. 把模型输出的 Tensor 动作转回 Numpy (包含在计时内)
            action_np = action.squeeze(0).cpu().numpy()

            # ⏱️ 结束计时
            inference_time_ms = (time.perf_counter() - start_time) * 1000 # 转换为毫秒 (ms)
            ep_inference_times.append(inference_time_ms)
            # print(f"👉 Step {step} 推理耗时: {inference_time_ms:.2f} ms")
            # ==========================================
            try:
                # 5. 与环境交互
                obs, reward, terminated, truncated, info = env.step(action_np)
                ep_reward += float(reward)

                done = terminated or truncated
            except Exception as e:
                # 🌟 核心拦截器：无论物理引擎报什么错，全部强行吃掉！
                logging.error(f"物理引擎崩溃 (Step {step}): {e}")
                logging.error("判定本回合评估失败，直接结束当前回合，继续训练...")
                
                done = True
                ep_reward = -1000.0          # 发生物理崩溃，不给奖励
                info = {"is_success": False} # 标记为失败
            if done:
                break
        # 记录指标
        successes.append(info.get("is_success", False))
        rewards.append(ep_reward)
        episode_records.append(
            {
                "episode": ep,
                "seed": ep_seed,
                "success": bool(info.get("is_success", False)),
                "reward": float(ep_reward),
                "steps": int(steps_taken),
            }
        )

        # 6. 根据配置的帧率和最大渲染数量保存视频
        if ep < max_rendered and frames_by_camera:
            status = "Success" if successes[ep] else "Fail"
            for camera, frames in frames_by_camera.items():
                if len(frames) == 0:
                    continue
                video_name = f"{camera}_ep_{ep}_reward={ep_reward:.1f}_{status}.mp4"
                video_path = videos_dir / video_name
                imageio.mimsave(str(video_path), frames, fps=fps)
                saved_video_paths.append(str(video_path))

        # 记录每回合的推理时间
        if len(ep_inference_times) > 0:
            
            # 过滤出“真实推理”步骤（比如耗时超过 5ms 的肯定是在跑网络，排除了 0.1ms 的出队操作）
            real_inferences = [t for t in ep_inference_times if t > 5.0]
            
            if real_inferences:
                global_real_inference_times.extend(real_inferences) # 加入全局统计

    # 结算全局指标：剔除前几次全局预热     
    if len(global_real_inference_times) > 0:

        # 全局剔除前 3 次真正的网络推理作为 Warm-up
        warmup_steps = 3
        # 尝试获取稳定状态的数据
        if len(global_real_inference_times) > warmup_steps:
            stable_times = global_real_inference_times[warmup_steps:]
        else:
            # 数据太少，不够剔除，只能全量使用
            stable_times = global_real_inference_times
            
        # 安全地计算最大值和均值
        max_time = max(stable_times)

        avg_real_time = sum(stable_times) / len(stable_times)
        # logging.info(f"[总计{n_episodes}回合] 真实推理触发: {len(global_real_inference_times)} 次 | 峰值耗时: {max_time:.2f} ms | 平均耗时: {avg_real_time:.2f} ms")
    else:
        avg_real_time = 0.0  # 提供默认值防报错
        max_time = 0.0
    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)),
            "average_reward": float(np.mean(rewards)),
            "avg_inference_ms": float(avg_real_time),
            "max_inference_ms": float(max_time)
        },
        "video_paths": saved_video_paths,
        "episodes": episode_records,
    }


def seed_everything(seed: int, deterministic: bool = False):
    """Set seed for absolute reproducibility."""
    
    # 1. 锁死 Python 字典和集合的哈希种子 
    # (防止字典遍历顺序在不同运行中发生变化，导致 Batch 数据错位)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. 锁死 Python / Numpy / PyTorch (CPU & 所有 GPU)
    seed_runtime(seed)

    configure_torch_runtime(deterministic)

def ensure_python_hash_seed(seed: int):
    """PYTHONHASHSEED 必须在解释器启动前生效；独立运行时自动重启一次补齐。"""
    desired = str(seed)
    if os.environ.get("PYTHONHASHSEED") == desired:
        return

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = desired
    logging.warning(f"PYTHONHASHSEED 需要在 Python 启动前设置，正在用 PYTHONHASHSEED={desired} 自动重启 eval_policy.py。")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

def resolve_eval_path(path_like) -> Path:
    path = Path(path_like).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        root_path = ROOT_PATH / path
        if root_path.exists():
            return root_path.resolve()
        return root_path.resolve()
    return path


def checkpoint_step(checkpoint_path: Path) -> int:
    match = re.match(r"^(\d+)", checkpoint_path.name)
    return int(match.group(1)) if match else -1


def is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "pretrained_model").is_dir()
        or (path / "config.yaml").exists()
        or (path / "training_state.pth").exists()
    )


def resolve_record_checkpoint_path(raw_path, checkpoints_dir: Path) -> Path:
    raw = Path(str(raw_path)).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        Path.cwd() / raw,
        ROOT_PATH / raw,
        checkpoints_dir / raw,
        checkpoints_dir / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT_PATH / raw).resolve()


def unique_paths(paths: list[Path]) -> list[Path]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path.resolve())
    return unique


def discover_checkpoints(path_like, checkpoint_source: str = "all") -> tuple[list[Path], Path, bool]:
    """支持输入单个 checkpoint、checkpoints 目录或完整训练 run 目录。"""
    input_path = resolve_eval_path(path_like)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ 找不到评估路径: {input_path}\n请检查 ckpt_path 是否正确。")

    if is_checkpoint_dir(input_path):
        run_dir = input_path.parent.parent if input_path.parent.name == "checkpoints" else input_path.parent
        return [input_path], run_dir.resolve(), False

    if input_path.name == "checkpoints":
        checkpoints_dir = input_path
        run_dir = input_path.parent
    else:
        checkpoints_dir = input_path / "checkpoints"
        run_dir = input_path

    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"❌ 未找到 checkpoints 目录: {checkpoints_dir}\n"
            "ckpt_path 可以指向单个 checkpoint、checkpoints 目录，或训练输出 run 目录。"
        )

    checkpoint_source = checkpoint_source.lower()
    if checkpoint_source == "all":
        checkpoints = [p.resolve() for p in checkpoints_dir.iterdir() if is_checkpoint_dir(p)]
        checkpoints.sort(key=lambda p: (checkpoint_step(p), p.name))
    elif checkpoint_source in {"top_k", "latest"}:
        records_path = checkpoints_dir / "top_k_records.json"
        if not records_path.exists():
            raise FileNotFoundError(f"❌ {checkpoint_source} 需要 top_k_records.json，但未找到: {records_path}")

        with open(records_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if checkpoint_source == "latest":
            raw_paths = [records["latest"]]
        else:
            raw_paths = [item["path"] for item in records.get("top_k", [])]

        checkpoints = [resolve_record_checkpoint_path(raw_path, checkpoints_dir) for raw_path in raw_paths]
        missing = [str(path) for path in checkpoints if not is_checkpoint_dir(path)]
        if missing:
            raise FileNotFoundError("top_k_records.json 中存在不可用 checkpoint:\n" + "\n".join(missing))
    else:
        raise ValueError("checkpoint_source 只能是 all / top_k / latest")

    checkpoints = unique_paths(checkpoints)
    if not checkpoints:
        raise FileNotFoundError(f"❌ 没有在目录中发现可评估 checkpoint: {checkpoints_dir}")

    return checkpoints, run_dir.resolve(), True


def build_batch_output_dir(eval_cfg, run_dir: Path) -> Path:
    output_dir = getattr(eval_cfg, "eval_output_dir", None)
    if output_dir:
        output_path = Path(output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = ROOT_PATH / output_path
        return output_path.resolve()

    source = getattr(eval_cfg, "checkpoint_source", "all")
    mode = getattr(eval_cfg, "mode", "fast_repro")
    precision = "amp" if getattr(eval_cfg, "use_amp", False) and mode != "strict" else "fp32"
    return (
        run_dir
        / "policy_eval"
        / f"{source}_{mode}_{precision}_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}_steps={eval_cfg.max_steps}"
    ).resolve()


def make_checkpoint_eval_cfg(eval_cfg, checkpoint_path: Path):
    checkpoint_cfg = SimpleNamespace(**vars(eval_cfg))
    checkpoint_cfg.ckpt_path = str(checkpoint_path)
    if isinstance(getattr(eval_cfg, "render_camera", None), list):
        checkpoint_cfg.render_camera = list(eval_cfg.render_camera)
    return checkpoint_cfg


def checkpoint_identity_keys(checkpoint_path: Path) -> set[str]:
    checkpoint_path = checkpoint_path.resolve()
    return {str(checkpoint_path), checkpoint_path.name}


def row_identity_keys(row: dict) -> set[str]:
    keys = set()
    for field in ("checkpoint_key", "checkpoint_path", "checkpoint_name"):
        value = row.get(field)
        if not value:
            continue
        keys.add(str(value))
        if field == "checkpoint_path":
            try:
                keys.add(str(resolve_eval_path(value)))
            except Exception:
                pass
    return keys


def load_existing_eval_summary(summary_dir: Path) -> list[dict]:
    summary_path = summary_dir / "policy_eval_summary.json"
    if not summary_path.exists():
        return []

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logging.warning(f"读取已有评估汇总失败，将从空表继续: {summary_path} ({exc})")
        return []

    rows = payload.get("results", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        logging.warning(f"已有评估汇总格式异常，将从空表继续: {summary_path}")
        return []

    rows = clean_eval_rows(rows)
    logging.info(f"已读取已有评估汇总: {summary_path}，包含 {len(rows)} 条历史结果")
    return rows


def clean_eval_rows(rows: list[dict]) -> list[dict]:
    cleaned_rows = []
    for row in rows:
        cleaned = dict(row)
        cleaned.pop("eval_order", None)
        cleaned.pop("completed_at", None)
        cleaned_rows.append(cleaned)
    return cleaned_rows


def completed_checkpoint_keys(rows: list[dict]) -> set[str]:
    keys = set()
    for row in rows:
        if row.get("status") != "ok":
            continue
        keys.update(row_identity_keys(row))
    return keys


def latest_rows_by_checkpoint(rows: list[dict]) -> list[dict]:
    latest = {}
    fallback_index = 0
    for row in rows:
        keys = row_identity_keys(row)
        if not keys:
            fallback_index += 1
            key = f"__row_{fallback_index}"
        else:
            key = sorted(keys)[0]
        latest[key] = row
    return list(latest.values())


def ranked_eval_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    latest_rows = latest_rows_by_checkpoint(rows)
    ok_rows = [row for row in latest_rows if row.get("status") == "ok"]
    failed_rows = [row for row in latest_rows if row.get("status") != "ok"]
    sorted_ok_rows = sorted(
        ok_rows,
        key=lambda row: (row.get("success_rate", -1.0), row.get("average_reward", float("-inf"))),
        reverse=True,
    )
    ranked_rows = []
    for rank, row in enumerate(sorted_ok_rows, start=1):
        ranked_row = dict(row)
        ranked_row["rank"] = rank
        ranked_rows.append(ranked_row)
    for row in failed_rows:
        ranked_row = dict(row)
        ranked_row["rank"] = None
        ranked_rows.append(ranked_row)
    return latest_rows, ok_rows, ranked_rows


def write_eval_summary(summary_dir: Path, rows: list[dict], eval_cfg, source_path) -> tuple[Path, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = clean_eval_rows(rows)
    latest_rows, ok_rows, ranked_rows = ranked_eval_rows(rows)

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_path": str(source_path),
        "checkpoint_source": getattr(eval_cfg, "checkpoint_source", "all"),
        "mode": getattr(eval_cfg, "mode", "fast_repro"),
        "seed": eval_cfg.seed,
        "n_episodes": eval_cfg.n_episodes,
        "max_steps": eval_cfg.max_steps,
        "total_records": len(rows),
        "latest_checkpoint_records": len(latest_rows),
        "best": ranked_rows[0] if ok_rows else None,
        "ranking": ranked_rows,
        "results": rows,
    }
    json_path = summary_dir / "policy_eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = summary_dir / "policy_eval_summary.csv"
    fieldnames = [
        "rank",
        "status",
        "checkpoint_name",
        "checkpoint_key",
        "step",
        "success_rate",
        "success_rate_percent",
        "average_reward",
        "avg_inference_ms",
        "max_inference_ms",
        "n_episodes",
        "max_steps",
        "seed",
        "mode",
        "videos_dir",
        "episode_results_path",
        "checkpoint_path",
        "error",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranked_rows, start=1):
            csv_row = {field: row.get(field, "") for field in fieldnames}
            csv_row["rank"] = rank if row.get("status") == "ok" else ""
            writer.writerow(csv_row)

    logging.info(f"批量评估汇总 JSON 已保存: {json_path}")
    logging.info(f"批量评估汇总 CSV 已保存: {csv_path}")
    return json_path, csv_path


def prune_ranked_eval_artifacts(
    run_dir: Path,
    summary_dir: Path,
    rows: list[dict],
    keep_top: int,
    prune_checkpoints: bool = True,
    prune_eval_outputs: bool = True,
    prune_unranked_checkpoints: bool = False,
) -> dict:
    if keep_top <= 0:
        raise ValueError("keep_top_after_eval 必须大于 0，或者设为 None 关闭清理。")

    latest_rows, _, ranked_rows = ranked_eval_rows(rows)
    keep_rows = [row for row in ranked_rows if row.get("status") == "ok"][:keep_top]
    keep_names = {row.get("checkpoint_name") for row in keep_rows if row.get("checkpoint_name")}
    evaluated_names = {row.get("checkpoint_name") for row in latest_rows if row.get("checkpoint_name")}

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keep_top": keep_top,
        "kept_checkpoints": [row.get("checkpoint_name") for row in keep_rows],
        "deleted_checkpoints": [],
        "deleted_eval_outputs": [],
        "skipped_unranked_checkpoints": [],
        "skipped": [],
    }

    if not keep_names:
        logging.warning("没有成功评估的 checkpoint，跳过按排名清理。")
        report["skipped"].append("no_successful_checkpoint")
        return report

    if prune_checkpoints:
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            for checkpoint_dir in checkpoints_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue
                if checkpoint_dir.name in keep_names:
                    continue
                if checkpoint_dir.name not in evaluated_names and not prune_unranked_checkpoints:
                    report["skipped_unranked_checkpoints"].append(str(checkpoint_dir))
                    continue
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                report["deleted_checkpoints"].append(str(checkpoint_dir))
                logging.info(f"已按评估排名删除低排名模型: {checkpoint_dir}")
        else:
            report["skipped"].append(f"missing_checkpoints_dir:{checkpoints_dir}")

    if prune_eval_outputs and summary_dir.exists():
        for eval_dir in summary_dir.iterdir():
            if not eval_dir.is_dir() or eval_dir.name not in evaluated_names:
                continue
            if eval_dir.name in keep_names:
                continue
            shutil.rmtree(eval_dir, ignore_errors=True)
            report["deleted_eval_outputs"].append(str(eval_dir))
            logging.info(f"已按评估排名删除低排名评估输出: {eval_dir}")

    report_path = summary_dir / "policy_eval_prune_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logging.info(f"评估清理报告已保存: {report_path}")
    return report


def prune_if_configured(
    eval_cfg,
    run_dir: Path,
    batch_output_dir: Path | None,
    rows: list[dict],
    stage: str,
    prune_unranked_checkpoints: bool = False,
):
    if batch_output_dir is None:
        return None

    keep_top_after_eval = getattr(eval_cfg, "keep_top_after_eval", None)
    if keep_top_after_eval is None:
        return None

    if getattr(eval_cfg, "max_checkpoints", None) and not getattr(eval_cfg, "allow_prune_with_max_checkpoints", False):
        logging.warning("当前设置了 max_checkpoints，跳过清理，避免删除尚未完整评估的 checkpoint。")
        return None

    logging.info(f"{stage}: 按 ranking 清理评估产物，仅保留前 {int(keep_top_after_eval)} 个 checkpoint。")
    return prune_ranked_eval_artifacts(
        run_dir=run_dir,
        summary_dir=batch_output_dir,
        rows=rows,
        keep_top=int(keep_top_after_eval),
        prune_checkpoints=bool(getattr(eval_cfg, "prune_checkpoints", True)),
        prune_eval_outputs=bool(getattr(eval_cfg, "prune_eval_outputs", True)),
        prune_unranked_checkpoints=prune_unranked_checkpoints,
    )


def evaluate_one_checkpoint(eval_cfg, checkpoint_path: Path, device, output_root: Path | None = None) -> dict:
    ckpt_path = checkpoint_path.resolve()
    eval_mode = getattr(eval_cfg, "mode", "fast_repro")
    eval_env = None
    policy = None

    try:
        # ==========================================
        # 1.自动探测 LeRobot 的 pretrained_model 子文件夹
        # ==========================================
        hf_model_dir = ckpt_path / "pretrained_model"
        if hf_model_dir.exists():
            logging.info("检测到 LeRobot 标准快照结构，将自动读取子目录: pretrained_model")
            load_dir = hf_model_dir
        else:
            load_dir = ckpt_path

        # ==========================================
        # 2.实例化 Policy 并加载权重
        # ==========================================
        logging.info(f"正在从目录重建网络并加载权重: {load_dir}")
        try:
            from lerobot.common.utils.utils import init_hydra_config

            config_yaml_path = Path(load_dir) / "config.yaml"
            if not config_yaml_path.exists():
                config_yaml_path = Path(load_dir).parent / "config.yaml"

            if not config_yaml_path.exists():
                raise FileNotFoundError("找不到 config.yaml，无法初始化 hydra_cfg！")

            hydra_cfg = init_hydra_config(str(config_yaml_path))
            hydra_cfg.device = str(device)

            policy = make_policy(
                hydra_cfg=hydra_cfg,
                pretrained_policy_name_or_path=str(load_dir)
            )

            logging.info("  成功使用 make_policy 加载策略！底层 Normalizer 与平滑权重已自动生效。")
            policy.to(device)
        except Exception as e:
            raise RuntimeError(f"❌ 权重加载失败！详细报错: {e}")

        # ==========================================
        # 🌟 3.读取快照中的配置，使环境和训练时的对齐
        # ==========================================
        all_obs_keys = policy.config.input_shapes.keys()
        ref_cams = [k.replace("observation.images.", "") for k in all_obs_keys if "observation.images." in k]
        if not ref_cams:
            raise ValueError("❌ 严重冲突：模型中未找到相机相关参数。请检查模型输入是否正确。")

        render_cameras = [eval_cfg.render_camera] if isinstance(eval_cfg.render_camera, str) else list(eval_cfg.render_camera)
        eval_cfg.render_camera = render_cameras
        obs_cameras = list(dict.fromkeys(ref_cams))

        config_yaml_path = Path(load_dir) / "config.yaml"
        if config_yaml_path.exists():
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f)

                env_cfg = full_cfg.get("env", {})
                env_name = env_cfg.get("name", getattr(env_cfg, "name", "guided_vision"))
                env_task = env_cfg.get("task", getattr(env_cfg, "task", "SewNeedle-3Arms-v0"))
                logging.info(f"成功从预训练文件夹读取完整环境配置: {env_name}/{env_task}")
        else:
            env_name = getattr(eval_cfg, "name", "guided_vision")
            env_task = getattr(eval_cfg, "task", "SewNeedle-3Arms-v0")
            logging.warning(f"  未找到 config.yaml，使用本地设定的后备环境: {env_name}/{env_task}")

        env_id = f"{env_name}/{env_task}"
        logging.info(f"正在通过 Gym 注册表构建环境: {env_id}")
        eval_env = gym.make(
            id=env_id,
            cameras=obs_cameras,
            episode_length=eval_cfg.max_steps
        )
        env_desc = f"{env_id} -> {eval_env.unwrapped.__class__.__module__}.{eval_env.unwrapped.__class__.__name__}"
        logging.info(f"当前评估环境: {env_desc}")
        logging.info(f"环境加载成功！最终挂载的相机: {obs_cameras}")

        # ==========================================
        # 🌟 4.设置视频和明细输出目录
        # ==========================================
        if output_root is None:
            videos_dir = ckpt_path / "extra_eval_videos"
        else:
            videos_dir = Path(output_root)
        logging.info(f"开始测试! 录像和明细将保存在: {videos_dir}")

        # ==========================================
        # 🌟 5.调用评估函数
        # ==========================================
        with torch.autocast(device_type=device.type) if getattr(eval_cfg, "use_amp", False) else nullcontext():
            eval_info = custom_eval_policy(
                env=eval_env,
                policy=policy,
                cfg_eval=eval_cfg,
                videos_dir=videos_dir,
                device=device
            )

        # ==========================================
        # 🌟 6.整理指标，归档视频和逐 episode 明细
        # ==========================================
        sr = eval_info["aggregated"]["success_rate"]
        ar = eval_info["aggregated"]["average_reward"]
        avg_infer = eval_info["aggregated"]["avg_inference_ms"]
        max_infer = eval_info["aggregated"]["max_inference_ms"]
        new_folder_name = f"eval_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}_sr={sr*100:.1f}_ar={ar:.2f}"
        new_videos_dir = Path(videos_dir) / new_folder_name
        new_videos_dir.mkdir(parents=True, exist_ok=True)

        for video_path in eval_info["video_paths"]:
            video_path = Path(video_path)
            if video_path.exists():
                shutil.move(str(video_path), str(new_videos_dir / video_path.name))

        episode_results_path = new_videos_dir / "episode_results.json"
        with open(episode_results_path, "w", encoding="utf-8") as f:
            json.dump(eval_info["episodes"], f, indent=2, ensure_ascii=False)
        logging.info(f"逐 episode 评估明细已保存: {episode_results_path}")

        result = {
            "status": "ok",
            "checkpoint_name": ckpt_path.name,
            "checkpoint_key": str(ckpt_path),
            "checkpoint_path": str(ckpt_path),
            "step": checkpoint_step(ckpt_path),
            "mode": eval_mode,
            "seed": eval_cfg.seed,
            "n_episodes": eval_cfg.n_episodes,
            "max_steps": eval_cfg.max_steps,
            "success_rate": float(sr),
            "success_rate_percent": float(sr * 100.0),
            "average_reward": float(ar),
            "avg_inference_ms": float(avg_infer),
            "max_inference_ms": float(max_infer),
            "videos_dir": str(new_videos_dir),
            "episode_results_path": str(episode_results_path),
            "error": "",
        }

        logging.info("="*50)
        logging.info("--独立评估完成！")
        logging.info(f"--Checkpoint: {ckpt_path.name}")
        logging.info(f"--评估模式: {eval_mode}")
        logging.info(f"--成功率 (Success Rate): {sr*100:.1f}%")
        logging.info(f"--平均奖励 (Average Reward): {ar:.2f}")
        logging.info(f"--平均推理时间 (Average Inference Time): {avg_infer:.2f} ms")
        logging.info(f"--最大推理时间 (Max Inference Time): {max_infer:.2f} ms")
        logging.info("="*50)
        return result
    finally:
        if eval_env is not None:
            eval_env.close()
        del policy
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()


def main(eval_cfg):
    eval_mode = getattr(eval_cfg, "mode", "fast_repro")
    if eval_mode not in {"fast_repro", "strict"}:
        raise ValueError(f"未知评估模式: {eval_mode}. 可选: fast_repro / strict")

    deterministic = eval_mode == "strict" or bool(getattr(eval_cfg, "deterministic", DETERMINISTIC_EVAL))
    if deterministic and getattr(eval_cfg, "use_amp", False):
        logging.warning("strict/deterministic 模式下自动关闭 AMP，避免混合精度带来的数值差异。")
        eval_cfg.use_amp = False

    seed_everything(eval_cfg.seed, deterministic=deterministic)
    if deterministic:
        patch_act_position_embedding_for_determinism()

    checkpoint_source = getattr(eval_cfg, "checkpoint_source", "all")
    checkpoints, run_dir, is_batch_input = discover_checkpoints(eval_cfg.ckpt_path, checkpoint_source)

    max_checkpoints = getattr(eval_cfg, "max_checkpoints", None)
    if max_checkpoints:
        checkpoints = checkpoints[: int(max_checkpoints)]

    device = get_safe_torch_device(getattr(eval_cfg, "device", "cuda"))
    logging.info(f"初始化评估程序... 使用设备: {device}")
    logging.info(f"将评估 {len(checkpoints)} 个 checkpoint")

    batch_output_dir = build_batch_output_dir(eval_cfg, run_dir) if is_batch_input else None
    if batch_output_dir is not None:
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"批量评估输出目录: {batch_output_dir}")

    rows = load_existing_eval_summary(batch_output_dir) if batch_output_dir is not None else []
    completed_keys = completed_checkpoint_keys(rows)
    if completed_keys:
        completed_rows = sum(1 for row in rows if row.get("status") == "ok")
        logging.info(f"已发现 {completed_rows} 条成功历史结果，将跳过对应 checkpoint")

    continue_on_error = bool(getattr(eval_cfg, "continue_on_error", len(checkpoints) > 1))
    for eval_order, checkpoint_path in enumerate(tqdm(checkpoints, desc="Evaluating checkpoints"), start=1):
        checkpoint_keys = checkpoint_identity_keys(checkpoint_path)
        if batch_output_dir is not None and checkpoint_keys & completed_keys:
            logging.info(f"[{eval_order}/{len(checkpoints)}] 跳过已评估 checkpoint: {checkpoint_path.name}")
            continue

        checkpoint_cfg = make_checkpoint_eval_cfg(eval_cfg, checkpoint_path)
        output_root = batch_output_dir / checkpoint_path.name if batch_output_dir is not None else None
        logging.info(f"[{eval_order}/{len(checkpoints)}] 开始评估 checkpoint: {checkpoint_path.name}")

        try:
            result = evaluate_one_checkpoint(
                eval_cfg=checkpoint_cfg,
                checkpoint_path=checkpoint_path,
                device=device,
                output_root=output_root,
            )
            rows.append(result)
            completed_keys.update(checkpoint_keys)
            if batch_output_dir is not None:
                write_eval_summary(batch_output_dir, rows, eval_cfg, source_path=resolve_eval_path(eval_cfg.ckpt_path))
                prune_if_configured(eval_cfg, run_dir, batch_output_dir, rows, stage="单个 checkpoint 评估完成")
        except Exception as exc:
            error_row = {
                "status": "failed",
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_key": str(checkpoint_path.resolve()),
                "checkpoint_path": str(checkpoint_path),
                "step": checkpoint_step(checkpoint_path),
                "mode": eval_mode,
                "seed": eval_cfg.seed,
                "n_episodes": eval_cfg.n_episodes,
                "max_steps": eval_cfg.max_steps,
                "error": repr(exc),
            }
            rows.append(error_row)
            if batch_output_dir is not None:
                write_eval_summary(batch_output_dir, rows, eval_cfg, source_path=resolve_eval_path(eval_cfg.ckpt_path))
                prune_if_configured(eval_cfg, run_dir, batch_output_dir, rows, stage="单个 checkpoint 评估失败")
            logging.exception(f"评估 checkpoint 失败: {checkpoint_path}")
            if not continue_on_error:
                raise

    if batch_output_dir is not None:
        write_eval_summary(batch_output_dir, rows, eval_cfg, source_path=resolve_eval_path(eval_cfg.ckpt_path))
        prune_if_configured(
            eval_cfg,
            run_dir,
            batch_output_dir,
            rows,
            stage="批量评估结束",
            prune_unranked_checkpoints=True,
        )
    return rows

# =========================================================================
# 🌟 独立评估测试入口，推荐使用lerobot保存的快照格式，只需要给路径，环境配置会自动对齐
# =========================================================================
if __name__ == "__main__":

    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    # ==========================================
    # 🎯 核心配置区：在这里自由修改你的评估参数！
    # ==========================================
    eval_cfg = SimpleNamespace(
        seed=100,
        # 可以指向单个 checkpoint，也可以指向整次训练 run 目录或 run/checkpoints 目录。
        ckpt_path="outputs/2_pretrain/train/2026-06-23/12-54-48_InsertCylinder-3Arms-v0_pre_wrist_diffusion",
        checkpoint_source="all",  # all: 读取目录下 checkpoint全部文件；   top_k/latest: 读取 checkpoints/top_k_records.json中记录的模型
        max_checkpoints=None,     # 调试时可设为 1/2，正式评估保持 None
        eval_output_dir=None,     # None 时自动保存到 run_dir/policy_eval/固定配置名，方便断点续评
        continue_on_error=True,   # 某个 checkpoint 失败时继续评估后面的模型
        keep_top_after_eval=10,   # 批量评估完成后只保留 ranking 前 N 个模型；设为 None 则不清理
        prune_checkpoints=True,   # 删除低排名的 checkpoints/模型文件夹
        prune_eval_outputs=True,  # 删除低排名的 policy_eval/当前评估输出文件夹，summary 会保留历史结果
        allow_prune_with_max_checkpoints=False,  # max_checkpoints 调试时默认不清理，避免误删未完整评估的模型
        
        # ⚙️ 评估参数设置
        mode="fast_repro",          # fast_repro: 快速且固定 seeds；  strict: 最强可复现但更慢
        n_episodes=100,             # 评估多少个任务                 
        max_episodes_rendered=0,    # 全量评估建议 0；需要视频时再改为 1/2
        fps=25,                     # 视频帧率，和环境控制频率对齐
        max_steps=400,              # 每个任务的最大步数
        device="cuda",              # 如需完全规避 CUDA 非确定算子，可临时改成 "cpu"
        deterministic=False,        # 通常不用手动改；strict 模式会自动开启
        
        # 相机设置
        # ['zed_cam_left', 'zed_cam_right', 'overhead_cam', 'worms_eye_cam' , 'wrist_cam_left', 'wrist_cam_right'],
        render_camera=['overhead_cam'],         # 保存video的相机视角    
        # ⚡ 快速评估默认开启混合精度；严格对比指标时可改 False
        use_amp=True,
    )
    if eval_cfg.deterministic or eval_cfg.mode == "strict" or DETERMINISTIC_EVAL:
        ensure_python_hash_seed(eval_cfg.seed)
    # ==========================================
    # 启动
    main(eval_cfg=eval_cfg)
                                         
