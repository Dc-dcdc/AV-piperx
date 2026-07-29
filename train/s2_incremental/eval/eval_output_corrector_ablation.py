#!/usr/bin/env python
"""Evaluate post-diffusion output-correction directions and strengths."""

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
    # 每项依次为：(名称, View→Arm输出scale, Arm→View输出scale)。
    ("arm_to_view_0", 0.0, 0.0),
    ("arm_to_view_0p25", 0.0, 0.25),
    ("arm_to_view_0p5", 0.0, 0.5),
    ("arm_to_view_0p75", 0.0, 0.75),
    ("arm_to_view_1", 0.0, 1.0),
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
    """Validate ``(name, View→Arm, Arm→View)`` output-scale presets."""
    if raw_presets is None:
        raw_presets = DEFAULT_ABLATION_PRESETS

    presets = []
    seen_names = set()
    for index, preset in enumerate(raw_presets):
        if not isinstance(preset, (tuple, list)) or len(preset) != 3:
            raise ValueError(
                "ablation_presets中的每一项必须是"
                "(name, view_to_arm_output_scale, arm_to_view_output_scale)，"
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
                _validate_scale("view_to_arm_output_scale", view_to_arm),
                _validate_scale("arm_to_view_output_scale", arm_to_view),
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
    """Separate ablation outputs by seed and episode count."""
    seed = int(seed)
    n_episodes = int(n_episodes)
    if n_episodes <= 0:
        raise ValueError(f"n_episodes必须为正整数，当前为{n_episodes}")

    base_output = (
        resolve_eval_path(requested)
        if requested
        else (checkpoint / "output_corrector_ablation").resolve()
    )
    eval_identity = f"eval_seed={seed}_ep={n_episodes}"
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
    """Clone common evaluation settings and apply one output-scale preset."""
    preset_cfg = SimpleNamespace(**vars(eval_cfg))
    preset_cfg.ckpt_path = str(checkpoint)
    preset_cfg.eval_output_dir = str(preset_output)
    preset_cfg.view_to_arm_output_scale = view_to_arm_scale
    preset_cfg.arm_to_view_output_scale = arm_to_view_scale

    # 这两个字段属于去噪瓶颈coupling，输出修正消融不得设置它们。
    preset_cfg.view_to_arm_coupling_scale = None
    preset_cfg.arm_to_view_coupling_scale = None

    if isinstance(getattr(eval_cfg, "render_camera", None), list):
        preset_cfg.render_camera = list(eval_cfg.render_camera)

    # 消融评估只读checkpoint，不允许排名逻辑删除任何训练或评估产物。
    preset_cfg.keep_top_after_eval = None
    preset_cfg.prune_checkpoints = False
    preset_cfg.prune_eval_outputs = False
    preset_cfg.allow_prune_with_max_checkpoints = False
    return preset_cfg


def _write_combined_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "output_corrector_ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    csv_path = output_root / "output_corrector_ablation_summary.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(eval_cfg) -> list[dict]:
    checkpoint = resolve_eval_path(eval_cfg.ckpt_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"评估路径不存在或不是目录: {checkpoint}")

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
        "将在同一评估输入上依次运行%d组最终输出修正消融，输出目录: %s",
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
                    "requested_view_to_arm_output_scale": view_to_arm_scale,
                    "requested_arm_to_view_output_scale": arm_to_view_scale,
                    **row,
                }
            )
        # 每完成一组立即保存，长时间评估中断后仍可保留已有结果。
        _write_combined_summary(output_root, combined_rows)

    return combined_rows


if __name__ == "__main__":
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    # ==========================================
    # 核心配置区：修改后直接运行本文件。
    # ==========================================
    eval_cfg = SimpleNamespace(
        seed=1400,  # 五组共用相同episode seeds，保证结果可以逐seed配对。
        # 可以指向单个checkpoint，也可以指向run或run/checkpoints目录。
        ckpt_path=(
            "outputs/2_pretrain/post_diffusion_output_corrector/InsertCylinder-3Arms-v0/2026-07-28/10-48-32_InsertCylinder-3Arms-v0_pre_zed_post_diffusion_output_corrector/checkpoints/045000_loss=0.0024_sr=87.0_ar=750.31"
        ),
        checkpoint_source="all",  # all / top_k / latest
        max_checkpoints=None,  # 调试时可设为1；正式评估保持None。
        # None时保存到checkpoint/output_corrector_ablation/eval_seed=<seed>_ep=<次数>。
        eval_output_dir=None,
        continue_on_error=False,

        # 每项格式：(名称, View→Arm output scale, Arm→View output scale)。
        # 当前默认保持View→Arm=0，只扫描论文所需的Arm→View强度。
        # None表示沿用checkpoint内该方向原始scale。
        ablation_presets=[
            ("arm_to_view_0", 0.0, 0.0),
            # ("arm_to_view_0p25", 0.0, 0.25),
            # ("arm_to_view_0p5", 0.0, 0.5),
            # ("arm_to_view_0p75", 0.0, 0.75),
            ("arm_to_view_1", 0.0, 1.0),
        ],

        # 评估参数，与eval_policy.py含义一致。
        mode="strict",  # fast_repro速度更快；最终论文结果可改为strict。
        n_episodes=200,
        max_episodes_rendered=0,  # 0不保存视频，只保存逐episode JSON。
        fps=25,
        max_steps=400,
        batch_size=15,
        use_async_envs=True,
        device="cuda:0",
        deterministic=False,
        render_camera=["zed_cam_left"],
        use_amp=True,

        # 消融脚本内部也会强制关闭所有清理操作。
        keep_top_after_eval=None,
        prune_checkpoints=False,
        prune_eval_outputs=False,
        allow_prune_with_max_checkpoints=False,
    )
    if eval_cfg.deterministic or eval_cfg.mode == "strict" or DETERMINISTIC_EVAL:
        ensure_python_hash_seed(eval_cfg.seed)
    main(eval_cfg=eval_cfg)
