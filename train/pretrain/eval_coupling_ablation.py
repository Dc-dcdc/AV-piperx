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

from train.pretrain.eval_policy import (
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


def _output_root(checkpoint: Path, requested: str | None) -> Path:
    if requested:
        return resolve_eval_path(requested)
    return (checkpoint / "coupling_ablation").resolve()


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
        seed=100,
        # 可以指向单个checkpoint，也可以指向run或run/checkpoints目录。
        ckpt_path=(
            "outputs/2_pretrain/train/2026-07-17/22-07-26_InsertCylinder-3Arms-v0_pre_zed_coupled_dual_head_diffusion/checkpoints/198000_loss=0.0020_sr=72.0_ar=680.02"
        ),
        checkpoint_source="all",  # all / top_k / latest
        max_checkpoints=None,      # 调试时可设为1；正式评测保持None
        # None时保存到ckpt_path/coupling_ablation；也可指定独立输出目录。
        eval_output_dir=None,
        continue_on_error=False,

        # 每项格式：(名称, View→Arm scale, Arm→View scale)。
        # None表示沿用checkpoint内对应方向的原始scale。
        ablation_presets=[
            # ("checkpoint_config", None, None),
            ("uncoupled", 0.0, 0.0),
            # ("arm_to_view_weak", 0, 0.25),
            # ("arm_to_view_trained", 0, 0.50),
            # ("arm_to_view_only", 0.0, 1.0),
            # ("view_to_arm_weak", 0.25, 0.0),
            # ("view_to_arm_trained", 0.5, 0.0),
            # ("view_to_arm_only", 1.0, 0.0),
            # ("weak_bidirectional", 0.25, 0.25),
            # ("medium_bidirectional", 0.5, 0.5),
            # ("strong_bidirectional", 1.0, 1.0),
        ],

        # 评测参数，与eval_policy.py含义一致。
        mode="fast_repro",       # fast_repro / strict
        n_episodes=100,
        max_episodes_rendered=0,
        fps=25,
        max_steps=400,
        batch_size=10,
        use_async_envs=True,
        device="cuda",
        deterministic=False,
        render_camera=["overhead_cam"],
        use_amp=True,

        # 为安全起见，消融脚本内部也会强制关闭所有清理操作。
        keep_top_after_eval=None,
        prune_checkpoints=False,
        prune_eval_outputs=False,
        allow_prune_with_max_checkpoints=False,
    )
    if eval_cfg.deterministic or eval_cfg.mode == "strict" or DETERMINISTIC_EVAL:
        ensure_python_hash_seed(eval_cfg.seed)
    main(eval_cfg=eval_cfg)
