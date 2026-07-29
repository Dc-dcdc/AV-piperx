#!/usr/bin/env python
"""Run paired directional/strength coupling ablations on one checkpoint input."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

from lerobot.common.utils.utils import init_logging

from train.s1_pretrain.eval.eval_policy import (
    DETERMINISTIC_EVAL,
    ensure_python_hash_seed,
    main as evaluate_policy,
    resolve_eval_path,
)


DEFAULT_ABLATION_PRESETS = (
    # None表示沿用checkpoint配置，而不是强制覆盖成1.0。
    ("checkpoint_config", None, None),
    ("arm_to_view_only", 0.0, 1.0),
    ("view_to_arm_only", 1.0, 0.0),
    ("uncoupled", 0.0, 0.0),
    ("weak_bidirectional", 0.25, 0.25),
    ("medium_bidirectional", 0.5, 0.5),
)


def _validate_scale(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}必须是None或[0, 1]内的有限数，当前为{value}")
    return value


def normalize_ablation_presets(
    raw_presets,
) -> list[tuple[str, float | None, float | None]]:
    """校验文件底部配置的(name, View→Arm, Arm→View)预设列表。"""
    if raw_presets is None:
        raw_presets = DEFAULT_ABLATION_PRESETS

    presets = []
    seen_names = set()
    for index, preset in enumerate(raw_presets):
        if not isinstance(preset, (tuple, list)) or len(preset) != 3:
            raise ValueError(
                "ablation_presets中的每一项必须是"
                "(name, view_to_arm_scale, arm_to_view_scale)，"
                f"第{index}项为{preset!r}"
            )

        raw_name, view_to_arm, arm_to_view = preset
        name = str(raw_name).strip()
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"消融名称必须是单层非空目录名，当前为{raw_name!r}")
        if name in seen_names:
            raise ValueError(f"消融名称不能重复，当前重复项为{name!r}")
        seen_names.add(name)

        presets.append(
            (
                name,
                _validate_scale("view_to_arm_coupling_scale", view_to_arm),
                _validate_scale("arm_to_view_coupling_scale", arm_to_view),
            )
        )

    if not presets:
        raise ValueError("ablation_presets不能为空")
    return presets


def _output_root(
    checkpoint: Path,
    requested: str | None,
    *,
    seed: int,
    n_episodes: int,
) -> Path:
    """按seed和episode数隔离评估目录，避免误复用其他评估配置的结果。"""
    seed = int(seed)
    n_episodes = int(n_episodes)
    if n_episodes <= 0:
        raise ValueError(f"n_episodes必须为正整数，当前为{n_episodes}")

    base_output = (
        resolve_eval_path(requested)
        if requested
        else (checkpoint / "coupling_ablation").resolve()
    )
    eval_identity = f"eval_seed={seed}_ep={n_episodes}"
    # 允许调用方直接传入已经带有相同标识的完整目录，避免重复嵌套。
    if base_output.name == eval_identity:
        return base_output
    return base_output / eval_identity


def make_preset_eval_cfg(
    eval_cfg,
    *,
    checkpoint: Path,
    preset_output: Path,
    view_to_arm_scale: float | None,
    arm_to_view_scale: float | None,
) -> SimpleNamespace:
    """复制公共评测配置，并只覆盖当前消融组需要变化的字段。"""
    preset_cfg = SimpleNamespace(**vars(eval_cfg))
    preset_cfg.ckpt_path = str(checkpoint)
    preset_cfg.eval_output_dir = str(preset_output)
    preset_cfg.view_to_arm_coupling_scale = view_to_arm_scale
    preset_cfg.arm_to_view_coupling_scale = arm_to_view_scale

    if isinstance(getattr(eval_cfg, "render_camera", None), list):
        preset_cfg.render_camera = list(eval_cfg.render_camera)

    # 消融评测不得因为某一组排名较低而删除checkpoint或其他组结果。
    preset_cfg.keep_top_after_eval = None
    preset_cfg.prune_checkpoints = False
    preset_cfg.prune_eval_outputs = False
    preset_cfg.allow_prune_with_max_checkpoints = False
    return preset_cfg


def _write_combined_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "coupling_ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    csv_path = output_root / "coupling_ablation_summary.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(eval_cfg) -> list[dict]:
    checkpoint = resolve_eval_path(eval_cfg.ckpt_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"评测路径不存在或不是目录: {checkpoint}")

    output_root = _output_root(
        checkpoint,
        getattr(eval_cfg, "eval_output_dir", None),
        seed=getattr(eval_cfg, "seed", 1000),
        n_episodes=getattr(eval_cfg, "n_episodes", 10),
    )
    presets = normalize_ablation_presets(
        getattr(eval_cfg, "ablation_presets", None)
    )
    combined_rows = []

    logging.info(
        "将在同一评测输入上依次运行%d组耦合消融，输出目录: %s",
        len(presets),
        output_root,
    )
    for index, (preset_name, view_to_arm_scale, arm_to_view_scale) in enumerate(
        presets,
        start=1,
    ):
        logging.info(
            "[%d/%d] 开始消融 %s: View→Arm=%s, Arm→View=%s",
            index,
            len(presets),
            preset_name,
            "checkpoint" if view_to_arm_scale is None else f"{view_to_arm_scale:g}",
            "checkpoint" if arm_to_view_scale is None else f"{arm_to_view_scale:g}",
        )
        preset_output = output_root / preset_name
        preset_cfg = make_preset_eval_cfg(
            eval_cfg,
            checkpoint=checkpoint,
            preset_output=preset_output,
            view_to_arm_scale=view_to_arm_scale,
            arm_to_view_scale=arm_to_view_scale,
        )
        result_rows = evaluate_policy(preset_cfg)
        for row in result_rows:
            combined_rows.append(
                {
                    "preset": preset_name,
                    "requested_view_to_arm_scale": view_to_arm_scale,
                    "requested_arm_to_view_scale": arm_to_view_scale,
                    **row,
                }
            )
        _write_combined_summary(output_root, combined_rows)

    return combined_rows


if __name__ == "__main__":
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    # ==========================================
    # 核心配置区：与eval_policy.py相同，直接在这里修改后运行本文件。
    # ==========================================
    eval_cfg = SimpleNamespace(
        seed=2000,
        # 可以指向单个checkpoint，也可以指向run或run/checkpoints目录。
        ckpt_path=(
            "outputs/2_pretrain/post_diffusion_output_corrector/InsertCylinder-3Arms-v0/2026-07-27/19-41-43_InsertCylinder-3Arms-v0_pre_zed_post_diffusion_output_corrector/checkpoints/010000_loss=0.0015_sr=86.0_ar=732.48"
        ),
        checkpoint_source="all",  # all / top_k / latest
        max_checkpoints=None,      # 调试时可设为1；正式评测保持None
        # None时保存到ckpt_path/coupling_ablation/eval_seed=<seed>_ep=<次数>；
        # 指定目录时也会自动追加相同的seed/ep子目录，避免复用其他配置的旧结果。
        eval_output_dir=None,
        continue_on_error=False,

        # 每项格式：(名称, View→Arm scale, Arm→View scale)。
        # None表示沿用checkpoint内对应方向的原始scale。
        ablation_presets=[
            # ("checkpoint_config", None, None),
            # ("uncoupled", 0.0, 0.0),
            # ("view_to_arm_0p5", 0.5, 0.0),
            ("arm_to_view_0p5", 0.0, 0.5),
            ("arm_to_view_0p25", 0.0, 0.25),
            ("arm_to_view_0p75", 0.0, 0.75),
            # ("bidirectional_0p5", 0.5, 0.5),
            # ("view_to_arm_0p1", 0.1, 0.0),
            # ("view_to_arm_0p25", 0.25, 0.0),

        ],

        # 评测参数，与eval_policy.py含义一致。
        mode="strict",       # fast_repro / strict
        n_episodes=200,
        max_episodes_rendered=0, # 0表示不渲染；>0表示渲染前N条episode
        fps=25,
        max_steps=400,
        batch_size=15,
        use_async_envs=True, # True时使用异步环境评测，False时使用同步环境评测。
        device="cuda:0",
        deterministic=True, # True时强制设置Python hash seed和PyTorch随机种子，保证每次评测结果完全一致。
        render_camera=["zed_cam_left"],
        use_amp=False, # True时使用半精度推理，False时使用单精度推理。

        # 为安全起见，消融脚本内部也会强制关闭所有清理操作。
        keep_top_after_eval=None,
        prune_checkpoints=False,
        prune_eval_outputs=False,
        allow_prune_with_max_checkpoints=False,
    )
    if eval_cfg.deterministic or eval_cfg.mode == "strict" or DETERMINISTIC_EVAL:
        ensure_python_hash_seed(eval_cfg.seed)
    main(eval_cfg=eval_cfg)
