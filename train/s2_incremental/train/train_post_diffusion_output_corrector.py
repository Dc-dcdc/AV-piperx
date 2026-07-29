#!/usr/bin/env python

"""Train a post-denoising 14x6 action-space corrector on a frozen dual policy.

The first run strictly migrates a raw ``dual_head_diffusion`` checkpoint into a
self-contained composite policy.  Deterministic baseline trajectory caches are
grouped by dataset/checkpoint/diffusion settings under ``outputs/buffer``, with
one ``seed=<sampling_seed>`` subdirectory per noise trajectory.  Later corrector
variants reuse matching caches and concatenate every configured seed for joint
shuffled training, without decoding images or running DDIM during optimization.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import sys
import types
import uuid
from bisect import bisect_right
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from lerobot.common.policies.diffusion.modeling_dual_head_diffusion import (
    DualHeadDiffusionPolicy,
)
from lerobot.common.policies.diffusion.modeling_post_diffusion_output_corrector import (
    PostDiffusionOutputCorrectorPolicy,
)
from lerobot.common.policies.factory import (
    _policy_cfg_from_hydra_cfg,
    get_policy_and_config_classes,
)
from lerobot.common.utils.utils import get_safe_torch_device
from train.s1_pretrain.train import train_pretrain as base_train

try:
    from .train_coupling_incremental import validate_baseline_config_compatibility
except ImportError:
    # 兼容 ``python train/s2_incremental/train/...py`` 直接启动方式。
    from train_coupling_incremental import validate_baseline_config_compatibility


_ORIGINAL_MAKE_POLICY = base_train.make_policy
_ORIGINAL_UPDATE_POLICY = base_train.update_policy
_ORIGINAL_LOAD_LOCAL_DATASET = base_train.load_local_lerobot_dataset
_ORIGINAL_REMOTE_DATASET_CLASS = base_train.LeRobotDataset
_CACHE_SCHEMA_VERSION = 1
_CACHE_GROUP_HASH_LENGTH = 12
_LEGACY_CACHE_HASH_LENGTH = 20
_GROUPED_CACHE_LAYOUT = "grouped_by_sampling_seed"
_ACTIVE_CACHE_MANIFESTS: tuple[Path, ...] = ()
_ACTIVE_CACHE_MEMORY_LIMIT_GB = 4.0


def _resolve_model_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    if not path.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {path}")
    return path


def _hash_file(hasher, path: Path) -> None:
    with open(path, "rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            hasher.update(chunk)


def checkpoint_fingerprint(model_dir: Path) -> str:
    """Hash the checkpoint files that determine frozen baseline behavior."""
    hasher = hashlib.sha256()
    candidates = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file()
        and (
            path.name in {"config.json", "config.yaml"}
            or path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
        )
    )
    if not candidates:
        raise FileNotFoundError(f"模型目录中没有可校验权重文件: {model_dir}")
    for path in candidates:
        hasher.update(path.name.encode("utf-8"))
        _hash_file(hasher, path)
    return hasher.hexdigest()


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        _plain(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _dataset_identity(
    cfg: DictConfig,
    dataset,
    delta_timestamps: dict,
) -> dict[str, Any]:
    local_dir = base_train.clean_optional_path(
        cfg.get("dataset_local_dir", None)
    )
    if local_dir is not None:
        local_path = Path(local_dir).expanduser().resolve()
        identity_files = [
            local_path / "meta_data" / "info.json",
            local_path / "meta_data" / "stats.safetensors",
            local_path / "meta_data" / "episode_data_index.safetensors",
            *sorted((local_path / "data").glob("*.parquet")),
        ]
        file_records = []
        for path in identity_files:
            stat = path.stat()
            file_records.append(
                {
                    "relative_path": str(path.relative_to(local_path)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        source = {"kind": "local", "path": str(local_path), "files": file_records}
    else:
        source = {"kind": "hub", "repo_id": str(cfg.dataset_repo_id)}

    return {
        "source": source,
        "num_samples": int(len(dataset)),
        "num_episodes": int(getattr(dataset, "num_episodes", 0)),
        "delta_timestamps": _plain(delta_timestamps),
    }


def _episode_index_to_json(dataset) -> dict[str, list[int]]:
    value = getattr(dataset, "episode_data_index", None)
    if value is None:
        return {}
    return {
        str(key): torch.as_tensor(child).cpu().tolist()
        for key, child in value.items()
    }


def _episode_index_from_json(value: dict[str, list[int]]) -> dict[str, Tensor]:
    return {
        key: torch.as_tensor(child, dtype=torch.long)
        for key, child in value.items()
    }


def _load_source_dataset(cfg: DictConfig, delta_timestamps: dict):
    """Load the image dataset only while constructing a missing cache."""
    local_dir = base_train.clean_optional_path(
        cfg.get("dataset_local_dir", None)
    )
    dataset_cache_dir = base_train.clean_optional_path(
        cfg.get("dataset_cache_dir", None)
    )
    if local_dir is not None:
        return _ORIGINAL_LOAD_LOCAL_DATASET(
            local_dir=local_dir,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.video_backend,
            image_transforms=None,
            cache_dir=dataset_cache_dir,
        )
    return _ORIGINAL_REMOTE_DATASET_CLASS(
        repo_id=cfg.dataset_repo_id,
        delta_timestamps=delta_timestamps,
        video_backend=cfg.video_backend,
        image_transforms=None,
    )


def _prepare_policy_inputs(policy, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    normalized = policy.normalize_inputs(batch)
    if policy.expected_image_keys:
        normalized = dict(normalized)
        normalized["observation.images"] = torch.stack(
            [normalized[key] for key in policy.expected_image_keys],
            dim=-4,
        )
    return normalized


def _sample_full_dual_trajectories(
    policy,
    normalized_batch: dict[str, Tensor],
    *,
    generator: torch.Generator,
) -> Tensor:
    diffusion = policy.diffusion
    global_cond = diffusion._prepare_global_conditioning(normalized_batch)
    batch_size = normalized_batch["observation.state"].shape[0]
    arm = diffusion.conditional_sample(
        diffusion.arm_unet,
        diffusion.arm_noise_scheduler,
        diffusion.arm_action_dim,
        batch_size,
        global_cond=global_cond,
        generator=generator,
    )
    view = diffusion.conditional_sample(
        diffusion.view_unet,
        diffusion.view_noise_scheduler,
        diffusion.view_action_dim,
        batch_size,
        global_cond=global_cond,
        generator=generator,
    )
    return torch.cat([arm, view], dim=-1)


def _cache_array_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "baseline": cache_dir / "baseline_action_trajectory.npy",
        "target": cache_dir / "target_action_trajectory.npy",
        "padding": cache_dir / "action_is_pad.npy",
    }


def _validate_cache_files(manifest_path: Path) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"缓存版本不兼容: {manifest.get('schema_version')} != "
            f"{_CACHE_SCHEMA_VERSION}"
        )
    cache_dir = manifest_path.parent
    for name, relative_path in manifest["arrays"].items():
        path = cache_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"缓存数组缺失({name}): {path}")
    return manifest


def _cache_key_payload(
    cfg: DictConfig,
    *,
    dataset_identity: dict[str, Any],
    source_fingerprint: str,
    sampling_seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "dataset": dataset_identity,
        "source_checkpoint_fingerprint": source_fingerprint,
        "sampling_seed": int(sampling_seed),
        "precompute_batch_size": int(cfg.output_cache.batch_size),
        "horizon": int(cfg.policy.horizon),
        "action_dim": int(cfg.policy.output_shapes.action[0]),
        "arm_action_dim": int(cfg.policy.arm_action_dim),
        "view_action_dim": int(cfg.policy.view_action_dim),
        "noise_scheduler_type": str(cfg.policy.noise_scheduler_type),
        "num_inference_steps": int(cfg.policy.num_inference_steps),
        "clip_sample": bool(cfg.policy.clip_sample),
        "clip_sample_range": float(cfg.policy.clip_sample_range),
        "storage_dtype": str(cfg.output_cache.storage_dtype),
    }


def resolve_output_cache_sampling_seeds(cfg: DictConfig) -> tuple[int, ...]:
    """Resolve the multi-seed list, with legacy single-seed compatibility."""
    configured_seeds = OmegaConf.select(
        cfg,
        "output_cache.sampling_seeds",
        default=None,
    )
    if configured_seeds is None:
        legacy_seed = OmegaConf.select(
            cfg,
            "output_cache.sampling_seed",
            default=None,
        )
        if legacy_seed is None:
            raise ValueError(
                "output_cache.sampling_seeds不能为空；旧配置也必须提供"
                "output_cache.sampling_seed。"
            )
        raw_seeds = [legacy_seed]
    else:
        raw_seeds = _plain(configured_seeds)
        if not isinstance(raw_seeds, list):
            raise TypeError("output_cache.sampling_seeds必须是整数列表。")

    if not raw_seeds:
        raise ValueError("output_cache.sampling_seeds至少需要一个随机种子。")

    seeds: list[int] = []
    for value in raw_seeds:
        if isinstance(value, bool):
            raise TypeError("sampling seed不能是布尔值。")
        try:
            seed = int(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"sampling seed必须是整数，当前值为{value!r}。"
            ) from error
        if isinstance(value, float) and not value.is_integer():
            raise TypeError(f"sampling seed不能是小数，当前值为{value!r}。")
        seeds.append(seed)

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"output_cache.sampling_seeds包含重复值: {seeds}"
        )
    return tuple(seeds)


def _cache_group_payload(
    key_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return the seed-independent identity shared by noise trajectories."""
    if "sampling_seed" not in key_payload:
        raise KeyError("缓存身份中缺少sampling_seed。")
    return {
        key: value
        for key, value in key_payload.items()
        if key != "sampling_seed"
    }


def _resolve_output_cache_layout(
    root: Path,
    key_payload: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the grouped location and its legacy flat-cache fallback."""
    cache_key = _stable_hash(key_payload)
    cache_group_key = _stable_hash(_cache_group_payload(key_payload))
    sampling_seed = int(key_payload["sampling_seed"])
    cache_group_dir = root / (
        f"dual_trajectory_{cache_group_key[:_CACHE_GROUP_HASH_LENGTH]}"
    )
    return {
        "cache_key": cache_key,
        "cache_group_key": cache_group_key,
        "cache_group_dir": cache_group_dir,
        "cache_dir": cache_group_dir / f"seed={sampling_seed}",
        "legacy_cache_dir": root
        / f"dual_trajectory_{cache_key[:_LEGACY_CACHE_HASH_LENGTH]}",
    }


def _manifest_cache_group_key(manifest: dict[str, Any]) -> str | None:
    """Read a new group key or reconstruct it from a legacy manifest."""
    if manifest.get("cache_group_key") is not None:
        return str(manifest["cache_group_key"])
    identity_fields = (
        "schema_version",
        "dataset",
        "source_checkpoint_fingerprint",
        "sampling_seed",
        "precompute_batch_size",
        "horizon",
        "action_dim",
        "arm_action_dim",
        "view_action_dim",
        "noise_scheduler_type",
        "num_inference_steps",
        "clip_sample",
        "clip_sample_range",
        "storage_dtype",
    )
    if any(field not in manifest for field in identity_fields):
        return None
    key_payload = {field: manifest[field] for field in identity_fields}
    return _stable_hash(_cache_group_payload(key_payload))


def _resolve_cache_source_model(cfg: DictConfig) -> Path:
    if bool(cfg.resume):
        resume_path = base_train.clean_optional_path(
            cfg.get("resume_path", None)
        )
        if resume_path is None:
            raise ValueError("resume=true但resume_path为空。")
        return _resolve_model_dir(resume_path)
    init_path = base_train.clean_optional_path(
        cfg.get("init_policy_path", None)
    )
    if init_path is None:
        raise ValueError("首次训练必须提供init_policy_path。")
    return _resolve_model_dir(init_path)


def _load_cache_source_policy(cfg: DictConfig, model_dir: Path):
    if bool(cfg.resume):
        policy = PostDiffusionOutputCorrectorPolicy.from_pretrained(
            model_dir,
            strict=True,
        )
    else:
        policy = DualHeadDiffusionPolicy.from_pretrained(
            model_dir,
            strict=True,
        )
    return policy


def build_or_reuse_output_cache(
    cfg: DictConfig,
    *,
    sampling_seed: int | None = None,
    source_dataset=None,
    dataset_identity: dict[str, Any] | None = None,
    source_model_dir: Path | None = None,
    source_fingerprint: str | None = None,
) -> Path:
    """Return a persistent manifest, creating its arrays only when missing."""
    if sampling_seed is None:
        sampling_seed = resolve_output_cache_sampling_seeds(cfg)[0]
    sampling_seed = int(sampling_seed)

    if source_dataset is None:
        delta_timestamps = base_train.get_resolved_delta_timestamps(cfg)
        source_dataset = _load_source_dataset(cfg, delta_timestamps)
    else:
        delta_timestamps = base_train.get_resolved_delta_timestamps(cfg)
    if dataset_identity is None:
        dataset_identity = _dataset_identity(
            cfg,
            source_dataset,
            delta_timestamps,
        )

    if source_model_dir is None:
        source_model_dir = _resolve_cache_source_model(cfg)
    if source_fingerprint is None:
        configured_fingerprint = OmegaConf.select(
            cfg,
            "output_cache.source_checkpoint_fingerprint",
            default=None,
        )
        if configured_fingerprint:
            source_fingerprint = str(configured_fingerprint)
        else:
            source_fingerprint = checkpoint_fingerprint(source_model_dir)

    key_payload = _cache_key_payload(
        cfg,
        dataset_identity=dataset_identity,
        source_fingerprint=source_fingerprint,
        sampling_seed=sampling_seed,
    )
    root = Path(cfg.output_cache.root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    layout = _resolve_output_cache_layout(root, key_payload)
    cache_key = layout["cache_key"]
    cache_group_key = layout["cache_group_key"]
    cache_group_dir = layout["cache_group_dir"]
    cache_dir = layout["cache_dir"]
    manifest_path = cache_dir / "manifest.json"
    legacy_manifest_path = layout["legacy_cache_dir"] / "manifest.json"
    rebuild = bool(getattr(cfg.output_cache, "rebuild", False))

    if manifest_path.is_file() and not rebuild:
        manifest = _validate_cache_files(manifest_path)
        if manifest["cache_key"] != cache_key:
            raise RuntimeError(
                "缓存目录命中但身份校验失败，拒绝复用: "
                f"{manifest_path}"
            )
        logging.info(
            "复用冻结dual轨迹缓存: %s (samples=%d, seed=%d)",
            manifest_path,
            manifest["num_samples"],
            manifest["sampling_seed"],
        )
    elif legacy_manifest_path.is_file() and not rebuild:
        # 旧版把seed写进顶层目录哈希。继续只读复用，避免复制大型数组；
        # 所有新建缓存统一使用“组哈希/seed=<seed>”目录。
        manifest = _validate_cache_files(legacy_manifest_path)
        if manifest["cache_key"] != cache_key:
            raise RuntimeError(
                "旧版缓存目录命中但身份校验失败，拒绝复用: "
                f"{legacy_manifest_path}"
            )
        manifest_path = legacy_manifest_path
        cache_dir = manifest_path.parent
        logging.info(
            "复用旧版扁平dual轨迹缓存: %s (samples=%d, seed=%d)；"
            "新缓存将使用dual_trajectory_<group>/seed=<seed>布局。",
            manifest_path,
            manifest["num_samples"],
            manifest["sampling_seed"],
        )
    else:
        if cache_dir.exists():
            # Preserve every prior/incomplete directory instead of deleting it.
            cache_dir = cache_group_dir / (
                f"seed={sampling_seed}_"
                f"rebuild_{uuid.uuid4().hex[:8]}"
            )
            manifest_path = cache_dir / "manifest.json"
        cache_dir.mkdir(parents=True, exist_ok=False)
        paths = _cache_array_paths(cache_dir)

        storage_dtype_name = str(cfg.output_cache.storage_dtype).lower()
        storage_dtype = {
            "float32": np.float32,
            "fp32": np.float32,
            "float16": np.float16,
            "fp16": np.float16,
        }.get(storage_dtype_name)
        if storage_dtype is None:
            raise ValueError(
                "output_cache.storage_dtype只支持float32或float16。"
            )

        num_samples = len(source_dataset)
        horizon = int(cfg.policy.horizon)
        action_dim = int(cfg.policy.output_shapes.action[0])
        baseline_array = np.lib.format.open_memmap(
            paths["baseline"],
            mode="w+",
            dtype=storage_dtype,
            shape=(num_samples, horizon, action_dim),
        )
        target_array = np.lib.format.open_memmap(
            paths["target"],
            mode="w+",
            dtype=storage_dtype,
            shape=(num_samples, horizon, action_dim),
        )
        padding_array = np.lib.format.open_memmap(
            paths["padding"],
            mode="w+",
            dtype=np.bool_,
            shape=(num_samples, horizon),
        )

        device = get_safe_torch_device(
            str(getattr(cfg.output_cache, "device", cfg.device)),
            log=True,
        )
        source_policy = _load_cache_source_policy(cfg, source_model_dir)
        source_policy.to(device)
        source_policy.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(sampling_seed)

        loader = DataLoader(
            source_dataset,
            batch_size=int(cfg.output_cache.batch_size),
            shuffle=False,
            num_workers=int(cfg.output_cache.num_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=bool(
                int(cfg.output_cache.num_workers) > 0
                and getattr(cfg.output_cache, "persistent_workers", True)
            ),
        )
        cursor = 0
        log_every = max(1, int(getattr(cfg.output_cache, "log_every_batches", 50)))
        logging.info(
            "开始构建冻结dual轨迹缓存: samples=%d, batch=%d, seed=%d, "
            "device=%s, output=%s",
            num_samples,
            int(cfg.output_cache.batch_size),
            sampling_seed,
            device,
            cache_dir,
        )
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                batch = {
                    key: (
                        value.to(device, non_blocking=True)
                        if isinstance(value, Tensor)
                        else value
                    )
                    for key, value in batch.items()
                }
                normalized_inputs = _prepare_policy_inputs(
                    source_policy,
                    batch,
                )
                baseline = _sample_full_dual_trajectories(
                    source_policy,
                    normalized_inputs,
                    generator=generator,
                )
                normalized_target = source_policy.normalize_targets(
                    {"action": batch["action"]}
                )["action"]
                padding = batch.get("action_is_pad")
                if padding is None:
                    padding = torch.zeros(
                        normalized_target.shape[:2],
                        dtype=torch.bool,
                        device=device,
                    )

                batch_count = int(baseline.shape[0])
                end = cursor + batch_count
                baseline_array[cursor:end] = (
                    baseline.detach().float().cpu().numpy().astype(storage_dtype)
                )
                target_array[cursor:end] = (
                    normalized_target.detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(storage_dtype)
                )
                padding_array[cursor:end] = (
                    padding.detach().bool().cpu().numpy()
                )
                cursor = end
                if batch_index % log_every == 0 or cursor == num_samples:
                    logging.info(
                        "缓存进度: %d/%d (%.1f%%)",
                        cursor,
                        num_samples,
                        100.0 * cursor / max(1, num_samples),
                    )

        if cursor != num_samples:
            raise RuntimeError(
                f"缓存样本数不完整: expected={num_samples}, actual={cursor}"
            )
        baseline_array.flush()
        target_array.flush()
        padding_array.flush()

        manifest = {
            **key_payload,
            "cache_key": cache_key,
            "cache_group_key": cache_group_key,
            "cache_layout": _GROUPED_CACHE_LAYOUT,
            "created_from_model_dir": str(source_model_dir),
            "num_samples": num_samples,
            "num_episodes": int(getattr(source_dataset, "num_episodes", 0)),
            "episode_data_index": _episode_index_to_json(source_dataset),
            "arrays": {
                name: path.name for name, path in paths.items()
            },
        }
        temporary_manifest = cache_dir / "manifest.json.tmp"
        with open(temporary_manifest, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
        os.replace(temporary_manifest, manifest_path)
        logging.info("冻结dual轨迹缓存已保存: %s", manifest_path)

        del source_policy, loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return manifest_path


def build_or_reuse_output_caches(cfg: DictConfig) -> tuple[Path, ...]:
    """Build/reuse every configured seed cache and return all manifests."""
    sampling_seeds = resolve_output_cache_sampling_seeds(cfg)
    delta_timestamps = base_train.get_resolved_delta_timestamps(cfg)
    source_dataset = _load_source_dataset(cfg, delta_timestamps)
    dataset_identity = _dataset_identity(
        cfg,
        source_dataset,
        delta_timestamps,
    )
    source_model_dir = _resolve_cache_source_model(cfg)
    configured_fingerprint = OmegaConf.select(
        cfg,
        "output_cache.source_checkpoint_fingerprint",
        default=None,
    )
    source_fingerprint = (
        str(configured_fingerprint)
        if configured_fingerprint
        else checkpoint_fingerprint(source_model_dir)
    )

    manifests = tuple(
        build_or_reuse_output_cache(
            cfg,
            sampling_seed=sampling_seed,
            source_dataset=source_dataset,
            dataset_identity=dataset_identity,
            source_model_dir=source_model_dir,
            source_fingerprint=source_fingerprint,
        )
        for sampling_seed in sampling_seeds
    )
    OmegaConf.update(
        cfg,
        "output_cache.resolved_manifests",
        [str(path) for path in manifests],
        force_add=True,
    )
    # 保留旧字段，方便既有日志、快照和外部脚本读取第一个缓存。
    OmegaConf.update(
        cfg,
        "output_cache.resolved_manifest",
        str(manifests[0]),
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "output_cache.source_checkpoint_fingerprint",
        source_fingerprint,
        force_add=True,
    )
    logging.info(
        "多seed轨迹缓存准备完成: seeds=%s, manifests=%d, samples=%d",
        list(sampling_seeds),
        len(manifests),
        len(source_dataset) * len(manifests),
    )
    return manifests


class CorrectionTrajectoryCacheDataset(Dataset):
    """Small training dataset backed by persistent NPY arrays or RAM."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        memory_limit_gb: float,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = _validate_cache_files(self.manifest_path)
        self.cache_dir = self.manifest_path.parent
        self.paths = {
            name: self.cache_dir / relative
            for name, relative in self.manifest["arrays"].items()
        }
        self.total_bytes = sum(
            path.stat().st_size for path in self.paths.values()
        )
        memory_limit_bytes = max(0.0, float(memory_limit_gb)) * 1024**3
        self.load_into_memory = self.total_bytes <= memory_limit_bytes
        self._arrays: dict[str, np.ndarray] | None = None
        self.num_samples = int(self.manifest["num_samples"])
        self.num_episodes = max(1, int(self.manifest["num_episodes"]))
        self.episode_data_index = _episode_index_from_json(
            self.manifest.get("episode_data_index", {})
        )
        self.repo_id = f"cached/{self.manifest['cache_key'][:12]}"
        logging.info(
            "修正器训练缓存: manifest=%s, size=%.3f GiB, mode=%s",
            self.manifest_path,
            self.total_bytes / 1024**3,
            "memory" if self.load_into_memory else "disk-mmap",
        )

    def _ensure_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            mmap_mode = None if self.load_into_memory else "r"
            self._arrays = {
                name: np.load(path, mmap_mode=mmap_mode)
                for name, path in self.paths.items()
            }
        return self._arrays

    def __getstate__(self):
        state = dict(self.__dict__)
        if not self.load_into_memory:
            state["_arrays"] = None
        return state

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        arrays = self._ensure_arrays()
        return {
            "baseline_action_trajectory": torch.from_numpy(
                np.array(arrays["baseline"][index], dtype=np.float32, copy=True)
            ),
            "target_action_trajectory": torch.from_numpy(
                np.array(arrays["target"][index], dtype=np.float32, copy=True)
            ),
            "action_is_pad": torch.from_numpy(
                np.array(arrays["padding"][index], dtype=np.bool_, copy=True)
            ),
        }


class CombinedCorrectionTrajectoryCacheDataset(Dataset):
    """Concatenate compatible noise-seed caches into one shuffled dataset."""

    def __init__(
        self,
        manifest_paths: list[str | Path] | tuple[str | Path, ...],
        *,
        memory_limit_gb: float,
    ) -> None:
        if not manifest_paths:
            raise ValueError("联合缓存数据集至少需要一个manifest。")
        resolved_paths = tuple(
            Path(path).expanduser().resolve() for path in manifest_paths
        )
        if len(set(resolved_paths)) != len(resolved_paths):
            raise ValueError("联合缓存数据集包含重复manifest。")

        manifests = [_validate_cache_files(path) for path in resolved_paths]
        resolved_group_keys = [
            _manifest_cache_group_key(manifest) for manifest in manifests
        ]
        if len(manifests) > 1 and any(
            group_key is None for group_key in resolved_group_keys
        ):
            raise ValueError(
                "存在无法验证缓存组身份的旧manifest，拒绝进行多seed联合训练。"
            )
        group_keys = {
            group_key
            for group_key in resolved_group_keys
            if group_key is not None
        }
        if len(group_keys) > 1:
            raise ValueError(
                "不能联合训练来自不同缓存组的数据: "
                f"{sorted(group_keys)}"
            )
        sampling_seeds = [
            int(manifest["sampling_seed"])
            for manifest in manifests
            if manifest.get("sampling_seed") is not None
        ]
        if len(manifests) > 1 and len(sampling_seeds) != len(manifests):
            raise ValueError(
                "存在缺少sampling_seed的manifest，拒绝进行多seed联合训练。"
            )
        if len(set(sampling_seeds)) != len(sampling_seeds):
            raise ValueError(
                f"联合缓存包含重复sampling_seed: {sampling_seeds}"
            )

        total_bytes = 0
        for manifest_path, manifest in zip(
            resolved_paths,
            manifests,
            strict=True,
        ):
            total_bytes += sum(
                (manifest_path.parent / relative_path).stat().st_size
                for relative_path in manifest["arrays"].values()
            )
        memory_limit_bytes = max(0.0, float(memory_limit_gb)) * 1024**3
        load_all_into_memory = total_bytes <= memory_limit_bytes
        child_memory_limit_gb = (
            float(memory_limit_gb) if load_all_into_memory else 0.0
        )

        self.datasets = tuple(
            CorrectionTrajectoryCacheDataset(
                manifest_path,
                memory_limit_gb=child_memory_limit_gb,
            )
            for manifest_path in resolved_paths
        )
        self.manifest_paths = resolved_paths
        self.manifests = tuple(manifests)
        self.sampling_seeds = tuple(sampling_seeds)
        self.total_bytes = total_bytes
        self.load_into_memory = load_all_into_memory
        self.cumulative_sizes: list[int] = []
        running_size = 0
        for dataset in self.datasets:
            running_size += len(dataset)
            self.cumulative_sizes.append(running_size)
        self.num_samples = running_size
        self.num_episodes = max(
            1,
            sum(dataset.num_episodes for dataset in self.datasets),
        )
        self.episode_data_index = self._merge_episode_data_indices()
        group_label = next(iter(group_keys), "legacy")
        self.repo_id = f"cached-multiseed/{group_label[:12]}"
        logging.info(
            "多seed联合训练数据集: seeds=%s, caches=%d, samples=%d, "
            "size=%.3f GiB, mode=%s",
            list(self.sampling_seeds),
            len(self.datasets),
            self.num_samples,
            self.total_bytes / 1024**3,
            "memory" if self.load_into_memory else "disk-mmap",
        )

    def _merge_episode_data_indices(self) -> dict[str, Tensor]:
        from_chunks: list[Tensor] = []
        to_chunks: list[Tensor] = []
        offset = 0
        for dataset in self.datasets:
            episode_index = dataset.episode_data_index
            if "from" not in episode_index or "to" not in episode_index:
                return {}
            from_chunks.append(episode_index["from"] + offset)
            to_chunks.append(episode_index["to"] + offset)
            offset += len(dataset)
        return {
            "from": torch.cat(from_chunks),
            "to": torch.cat(to_chunks),
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"联合缓存索引越界: index={index}, size={self.num_samples}"
            )
        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous_size = (
            0
            if dataset_index == 0
            else self.cumulative_sizes[dataset_index - 1]
        )
        return self.datasets[dataset_index][index - previous_size]


def _make_active_cache_dataset() -> Dataset:
    if not _ACTIVE_CACHE_MANIFESTS:
        raise RuntimeError("输出修正缓存尚未初始化。")
    if len(_ACTIVE_CACHE_MANIFESTS) == 1:
        return CorrectionTrajectoryCacheDataset(
            _ACTIVE_CACHE_MANIFESTS[0],
            memory_limit_gb=_ACTIVE_CACHE_MEMORY_LIMIT_GB,
        )
    return CombinedCorrectionTrajectoryCacheDataset(
        _ACTIVE_CACHE_MANIFESTS,
        memory_limit_gb=_ACTIVE_CACHE_MEMORY_LIMIT_GB,
    )


def _cached_local_dataset_loader(*args, **kwargs):
    del args, kwargs
    return _make_active_cache_dataset()


def _cached_remote_dataset_loader(*args, **kwargs):
    del args, kwargs
    return _make_active_cache_dataset()


def validate_training_config(cfg: DictConfig) -> None:
    if str(cfg.policy.name) != "post_diffusion_output_corrector":
        raise ValueError(
            "本入口只支持policy.name='post_diffusion_output_corrector'。"
        )
    if not bool(cfg.resume) and base_train.clean_optional_path(
        cfg.get("init_policy_path", None)
    ) is None:
        raise ValueError("首次训练必须提供原始dual checkpoint的init_policy_path。")
    if bool(cfg.resume) and base_train.clean_optional_path(
        cfg.get("init_policy_path", None)
    ) is not None:
        raise ValueError("resume=true时不能同时设置init_policy_path。")
    resolve_output_cache_sampling_seeds(cfg)
    if bool(cfg.training.image_transforms.enable):
        logging.warning(
            "输出修正器直接读取轨迹缓存，不使用图像增强；"
            "training.image_transforms.enable将被忽略。"
        )
    learning_rate = float(
        OmegaConf.select(
            cfg,
            "training.output_corrector_lr",
            default=cfg.training.lr,
        )
    )
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training.output_corrector_lr必须是有限正数。")

    if bool(getattr(cfg.training, "save_checkpoint", False)):
        save_freq = int(getattr(cfg.training, "save_freq", 0))
        eval_freq = int(getattr(cfg.training, "eval_freq", 0))
        if save_freq <= 0 or eval_freq <= 0 or save_freq != eval_freq:
            raise ValueError(
                "输出修正器要求每个checkpoint都完成评估，因此"
                "training.save_freq和training.eval_freq必须是相同的正整数；"
                f"当前save_freq={save_freq}, eval_freq={eval_freq}。"
            )
        if bool(getattr(cfg.eval, "async_enabled", False)) and bool(
            getattr(cfg.eval, "skip_if_busy", True)
        ):
            raise ValueError(
                "输出修正器要求每个checkpoint都完成评估；异步评估时必须设置"
                "eval.skip_if_busy=false，使队列繁忙时等待而不是跳过。"
            )


def migrate_dual_into_output_corrector(
    source_policy: DualHeadDiffusionPolicy,
    target_policy: PostDiffusionOutputCorrectorPolicy,
) -> dict[str, int]:
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    unexpected = sorted(set(source_state).difference(target_state))
    if unexpected:
        raise RuntimeError(f"目标复合策略无法识别基线状态: {unexpected}")

    shape_mismatches = [
        (
            key,
            tuple(source_tensor.shape),
            tuple(target_state[key].shape),
        )
        for key, source_tensor in source_state.items()
        if source_tensor.shape != target_state[key].shape
    ]
    if shape_mismatches:
        raise RuntimeError(f"dual→output-corrector形状不兼容: {shape_mismatches}")

    incompatible = target_policy.load_state_dict(source_state, strict=False)
    expected_missing = {
        key
        for key in target_state
        if key.startswith("diffusion.output_corrector.")
    }
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "不安全的dual→output-corrector迁移: "
            f"missing={incompatible.missing_keys}, "
            f"expected={sorted(expected_missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    migrated_state = target_policy.state_dict()
    unequal = [
        key
        for key, tensor in source_state.items()
        if not torch.equal(tensor, migrated_state[key])
    ]
    if unequal:
        raise RuntimeError(f"迁移后基线权重不再逐元素一致: {unequal}")
    return {
        "baseline_tensor_count": len(source_state),
        "corrector_tensor_count": len(expected_missing),
    }


def _global_condition_dim(policy) -> int:
    config = policy.config
    dimension = int(config.input_shapes["observation.state"][0])
    image_keys = [
        key
        for key in config.input_shapes
        if key.startswith("observation.image")
    ]
    if image_keys:
        dimension += (
            int(policy.diffusion.rgb_encoder.feature_dim) * len(image_keys)
        )
    if "observation.environment_state" in config.input_shapes:
        dimension += int(
            config.input_shapes["observation.environment_state"][0]
        )
    return dimension * int(config.n_obs_steps)


def run_scale_zero_equivalence_check(
    source_policy: DualHeadDiffusionPolicy,
    target_policy: PostDiffusionOutputCorrectorPolicy,
    *,
    seed: int,
) -> dict[str, float]:
    """Require bitwise equality for a complete fixed-noise dual trajectory."""
    source_was_training = source_policy.training
    target_was_training = target_policy.training
    source_policy.eval()
    target_policy.eval()
    device = next(target_policy.parameters()).device
    batch_size = 2
    global_generator = torch.Generator(device=device)
    global_generator.manual_seed(seed + 1)
    global_cond = torch.randn(
        batch_size,
        _global_condition_dim(target_policy),
        generator=global_generator,
        device=device,
    )

    source_generator = torch.Generator(device=device)
    source_generator.manual_seed(seed)
    target_generator = torch.Generator(device=device)
    target_generator.manual_seed(seed)
    with torch.inference_mode():
        source_arm = source_policy.diffusion.conditional_sample(
            source_policy.diffusion.arm_unet,
            source_policy.diffusion.arm_noise_scheduler,
            source_policy.diffusion.arm_action_dim,
            batch_size,
            global_cond=global_cond,
            generator=source_generator,
        )
        source_view = source_policy.diffusion.conditional_sample(
            source_policy.diffusion.view_unet,
            source_policy.diffusion.view_noise_scheduler,
            source_policy.diffusion.view_action_dim,
            batch_size,
            global_cond=global_cond,
            generator=source_generator,
        )
        target_arm, target_view = (
            target_policy.diffusion.generate_baseline_full_trajectories(
                batch_size,
                global_cond=global_cond,
                generator=target_generator,
            )
        )
        original_scales = (
            target_policy.diffusion.view_to_arm_output_scale,
            target_policy.diffusion.arm_to_view_output_scale,
        )
        target_policy.diffusion.set_output_correction_scales(
            view_to_arm=0.0,
            arm_to_view=0.0,
        )
        bypass_arm, bypass_view, _ = (
            target_policy.diffusion.apply_output_correction(
                target_arm,
                target_view,
            )
        )
        target_policy.diffusion.set_output_correction_scales(
            view_to_arm=original_scales[0],
            arm_to_view=original_scales[1],
        )

    arm_error = float((bypass_arm - source_arm).abs().max())
    view_error = float((bypass_view - source_view).abs().max())
    source_policy.train(source_was_training)
    target_policy.train(target_was_training)
    if not torch.equal(bypass_arm, source_arm) or not torch.equal(
        bypass_view,
        source_view,
    ):
        raise RuntimeError(
            "scale=0完整采样等价性检查失败: "
            f"arm_max_abs_error={arm_error:.3e}, "
            f"view_max_abs_error={view_error:.3e}"
        )
    return {
        "arm_max_abs_error": arm_error,
        "view_max_abs_error": view_error,
        "max_abs_error": max(arm_error, view_error),
    }


def configure_trainable_parameters(
    policy: PostDiffusionOutputCorrectorPolicy,
) -> dict[str, Any]:
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.diffusion.output_corrector.parameters():
        parameter.requires_grad_(True)
    trainable = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [
        name
        for name, _ in trainable
        if not name.startswith("diffusion.output_corrector.")
    ]
    if invalid:
        raise RuntimeError(f"发现非修正器可训练参数: {invalid}")
    if not trainable:
        raise RuntimeError("输出修正器没有可训练参数。")
    return {
        "trainable_names": [name for name, _ in trainable],
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in trainable
        ),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in policy.parameters()
            if not parameter.requires_grad
        ),
    }


def install_frozen_baseline_mode_guard(
    policy: PostDiffusionOutputCorrectorPolicy,
) -> None:
    frozen_modules = [
        policy.diffusion.arm_unet,
        policy.diffusion.view_unet,
    ]
    if getattr(policy.diffusion, "_use_images", False):
        frozen_modules.append(policy.diffusion.rgb_encoder)
    original_train = policy.train

    def guarded_train(self, mode: bool = True):
        result = original_train(mode)
        if mode:
            for module in frozen_modules:
                module.eval()
            self.diffusion.output_corrector.train(True)
        return result

    policy.train = types.MethodType(guarded_train, policy)
    policy.train(True)


def make_output_corrector_policy(
    hydra_cfg: DictConfig,
    pretrained_policy_name_or_path: str | None = None,
    dataset_stats=None,
    *,
    allow_scid_dual_init: bool = False,
    strict_pretrained_loading: bool = False,
):
    del allow_scid_dual_init, strict_pretrained_loading
    if pretrained_policy_name_or_path is None:
        raise ValueError(
            "输出修正器入口不允许随机初始化，必须使用init_policy_path或resume_path。"
        )
    if dataset_stats is not None:
        raise ValueError("从checkpoint加载时不应再次传入dataset_stats。")

    if bool(hydra_cfg.resume):
        policy = _ORIGINAL_MAKE_POLICY(
            hydra_cfg=hydra_cfg,
            pretrained_policy_name_or_path=pretrained_policy_name_or_path,
            dataset_stats=None,
            strict_pretrained_loading=True,
        )
        init_report: dict[str, Any] = {"mode": "resume"}
    else:
        policy_class, config_class = get_policy_and_config_classes(
            hydra_cfg.policy.name
        )
        if policy_class is not PostDiffusionOutputCorrectorPolicy:
            raise TypeError("目标Policy不是PostDiffusionOutputCorrectorPolicy。")
        target_config = _policy_cfg_from_hydra_cfg(config_class, hydra_cfg)
        source_policy = DualHeadDiffusionPolicy.from_pretrained(
            pretrained_policy_name_or_path,
            strict=True,
        )
        validate_baseline_config_compatibility(
            source_policy.config,
            target_config,
        )
        policy = PostDiffusionOutputCorrectorPolicy(target_config)
        migration = migrate_dual_into_output_corrector(source_policy, policy)

        device = get_safe_torch_device(hydra_cfg.device)
        source_policy.to(device)
        policy.to(device)
        equivalence = {}
        if bool(
            OmegaConf.select(
                hydra_cfg,
                "output_corrector.equivalence_check.enabled",
                default=True,
            )
        ):
            equivalence = run_scale_zero_equivalence_check(
                source_policy,
                policy,
                seed=int(
                    OmegaConf.select(
                        hydra_cfg,
                        "output_corrector.equivalence_check.seed",
                        default=20260727,
                    )
                ),
            )
        init_report = {
            "mode": "dual_checkpoint_migration",
            "migration": migration,
            "equivalence": equivalence,
        }
        del source_policy

    scope = configure_trainable_parameters(policy)
    install_frozen_baseline_mode_guard(policy)
    policy._output_corrector_init_report = init_report
    policy._output_corrector_scope_report = scope
    policy.to(get_safe_torch_device(hydra_cfg.device))
    logging.info(
        "输出修正器参数范围: trainable=%d tensors/%d params, frozen=%d params",
        scope["trainable_tensor_count"],
        scope["trainable_parameter_count"],
        scope["frozen_parameter_count"],
    )
    logging.info("可训练参数:\n%s", "\n".join(scope["trainable_names"]))
    if init_report.get("equivalence"):
        logging.info(
            "scale=0完整轨迹等价性检查通过: max_abs_error=%.3e",
            init_report["equivalence"]["max_abs_error"],
        )
    return policy


def _is_no_decay(name: str, parameter: nn.Parameter) -> bool:
    if parameter.ndim <= 1 or name.endswith(".bias"):
        return True
    return any(
        marker in name
        for marker in (
            ".arm_output.",
            ".view_output.",
            ".dimension_embedding",
        )
    )


def make_output_corrector_optimizer_and_scheduler(cfg: DictConfig, policy):
    trainable = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [
        name
        for name, _ in trainable
        if not name.startswith("diffusion.output_corrector.")
    ]
    if invalid:
        raise RuntimeError(f"优化器收到非修正器参数: {invalid}")
    learning_rate = float(
        OmegaConf.select(
            cfg,
            "training.output_corrector_lr",
            default=cfg.training.lr,
        )
    )
    structural_weight_decay = float(
        OmegaConf.select(
            cfg,
            "training.output_corrector_structural_weight_decay",
            default=1e-6,
        )
    )
    decay = [
        parameter
        for name, parameter in trainable
        if not _is_no_decay(name, parameter)
    ]
    no_decay = [
        parameter
        for name, parameter in trainable
        if _is_no_decay(name, parameter)
    ]
    groups = []
    if decay:
        groups.append(
            {
                "name": "output_corrector_structure",
                "params": decay,
                "lr": learning_rate,
                "weight_decay": structural_weight_decay,
            }
        )
    if no_decay:
        groups.append(
            {
                "name": "output_corrector_no_decay",
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=tuple(cfg.training.adam_betas),
        eps=float(cfg.training.adam_eps),
    )
    from diffusers.optimization import get_scheduler

    scheduler = get_scheduler(
        str(cfg.training.lr_scheduler),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=int(cfg.training.offline_steps),
    )
    logging.info(
        "输出修正器AdamW: lr=%g, structure=%d tensors(wd=%g), "
        "control/bias/norm=%d tensors(wd=0)",
        learning_rate,
        len(decay),
        structural_weight_decay,
        len(no_decay),
    )
    return optimizer, scheduler


def update_output_corrector_policy(*args, **kwargs):
    policy = args[0] if args else kwargs["policy"]
    policy.train(True)
    return _ORIGINAL_UPDATE_POLICY(*args, **kwargs)


def install_runtime_patches(
    manifest_paths: tuple[Path, ...],
    *,
    memory_limit_gb: float,
) -> None:
    global _ACTIVE_CACHE_MANIFESTS, _ACTIVE_CACHE_MEMORY_LIMIT_GB
    if not manifest_paths:
        raise ValueError("运行时至少需要一个轨迹缓存manifest。")
    _ACTIVE_CACHE_MANIFESTS = tuple(manifest_paths)
    _ACTIVE_CACHE_MEMORY_LIMIT_GB = float(memory_limit_gb)
    base_train.load_local_lerobot_dataset = _cached_local_dataset_loader
    base_train.LeRobotDataset = _cached_remote_dataset_loader
    base_train.make_policy = make_output_corrector_policy
    base_train.make_optimizer_and_scheduler = (
        make_output_corrector_optimizer_and_scheduler
    )
    base_train.update_policy = update_output_corrector_policy


def _save_effective_config(
    cfg: DictConfig,
    out_dir: str | Path,
) -> None:
    path = Path(out_dir) / ".hydra" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    logging.info("输出修正器实际配置已保存: %s", path)


@hydra.main(
    version_base="1.2",
    config_name="pre_default",
    config_path="../../../configs/pretrain",
)
def train_cli(cfg: DictConfig) -> None:
    hydra_runtime = hydra.core.hydra_config.HydraConfig.get()
    out_dir = hydra_runtime.run.dir
    # 严格续训以checkpoint配置为训练语义基线，但“评估队列忙时是否跳过”
    # 属于本次运行的调度策略。保留当前入口/default_args中的明确设置，
    # 避免旧快照里的skip_if_busy=true破坏每个checkpoint必评的约束。
    current_skip_if_busy = bool(
        OmegaConf.select(cfg, "eval.skip_if_busy", default=False)
    )
    cfg, resume_snapshot_config_path = base_train.build_resume_config(cfg)
    if resume_snapshot_config_path is not None:
        snapshot_cfg = OmegaConf.load(resume_snapshot_config_path)
        if OmegaConf.select(
            snapshot_cfg,
            "output_cache.sampling_seeds",
            default=None,
        ) is None:
            # 旧快照只有sampling_seed。避免当前配置文件中的默认多seed列表
            # 在严格续训时被当作缺省字段补入，从而悄悄改变训练数据分布。
            OmegaConf.update(
                cfg,
                "output_cache.sampling_seeds",
                None,
                merge=False,
                force_add=True,
            )
            logging.info(
                "旧版输出修正器快照未记录sampling_seeds；"
                "续训保持使用原sampling_seed=%s。",
                OmegaConf.select(
                    cfg,
                    "output_cache.sampling_seed",
                    default=None,
                ),
            )
    OmegaConf.update(
        cfg,
        "eval.skip_if_busy",
        current_skip_if_busy,
        merge=False,
        force_add=True,
    )
    validate_training_config(cfg)
    manifest_paths = build_or_reuse_output_caches(cfg)
    _save_effective_config(cfg, out_dir)
    install_runtime_patches(
        manifest_paths,
        memory_limit_gb=float(cfg.output_cache.memory_limit_gb),
    )
    base_train.train_dppo_pretrain(
        cfg,
        out_dir=out_dir,
        job_name=hydra_runtime.job.name,
    )


def _prepare_resume_cli(user_cli_args: tuple[str, ...]) -> None:
    is_resume = (
        str(base_train.get_cli_override_value(sys.argv, "resume")).lower()
        == "true"
    )
    resume_path = base_train.get_cli_override_value(sys.argv, "resume_path")
    if (
        not is_resume
        or not resume_path
        or resume_path.lower() in {"none", "null", ""}
    ):
        return

    base_train.replace_cli_override(sys.argv, "init_policy_path", "null")
    base_train.restore_resume_hydra_choices(
        sys.argv,
        user_cli_args,
        resume_path,
    )
    original_run_dir = base_train.get_resume_run_dir(resume_path)
    if original_run_dir is None:
        checkpoint_dir = base_train.get_resume_checkpoint_dir(resume_path)
        original_run_dir = checkpoint_dir.parent.parent
    base_train.replace_cli_override(
        sys.argv,
        "hydra.run.dir",
        f'"{original_run_dir.absolute()}"',
    )
    print(
        "🔄 [输出修正器恢复] 输出目录已重定向至原运行:\n"
        f"   👉 {original_run_dir.absolute()}"
    )


if __name__ == "__main__":
    explicit_cli_args = tuple(sys.argv[1:])
    default_args = [
        # 首次实验必须是原始dual-head checkpoint；不会读取其中的优化器状态。
        "init_policy_path='outputs/2_pretrain/train/2026-07-16/20-59-53_InsertCylinder-3Arms-v0_pre_zed_dual_head_diffusion/checkpoints/162000_loss=0.0051_sr=87.0_ar=751.50'",
        # 普通离线示范数据；首次构建缓存时读取图像，之后只读取轨迹数组。
        "dataset_local_dir=outputs/5_hf_datasets/quest_teleop_insert_cylinder_3arms_rgb_joint",
        # 本地数据不存在时使用的Hub仓库ID。
        "dataset_repo_id=Dc-dc/quest_teleop_insert_cylinder_3arms_rgb_joint",
        # 可跨实验复用的dual轨迹持久缓存根目录；命令行可直接覆盖。
        "output_cache.root_dir=outputs/buffer/post_diffusion_output_corrector",
        # 环境配置决定评估任务、状态维度和动作维度。
        "env=sim_insert_cylinder_3arms",
        # 14x6输出修正器的专用策略配置。
        "policy=pre_zed_post_diffusion_output_corrector",
        # false创建新实验；true从resume_path严格恢复复合策略与训练状态。
        "resume=false",
        # 首次保持null；断点续训时指向本实验checkpoint目录。
        "resume_path=null",
        # 轨迹缓存生成和普通训练DataLoader的CPU进程数。
        "training.num_workers=4",
        # false表示评估队列繁忙时等待，保证每个保存的checkpoint都被评估。
        "eval.skip_if_busy=false",
        # 是否上传训练、修正量和异步评估指标。
        "wandb.enable=true",
    ]
    for argument in default_args:
        key = argument.split("=", 1)[0]
        if base_train.get_cli_override_value(sys.argv, key) is None:
            sys.argv.append(argument)

    _prepare_resume_cli(explicit_cli_args)
    train_cli()
