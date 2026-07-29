#!/usr/bin/env python

"""Evaluate none, always Arm→View, and learned Router on identical seeds."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from lerobot.common.utils.utils import init_logging
from train.s1_pretrain.eval.eval_policy import (
    DETERMINISTIC_EVAL,
    ensure_python_hash_seed,
    main as evaluate_policy,
    resolve_eval_path,
)


DEFAULT_ROUTER_PRESETS = (
    ("none", "none"),
    ("always_arm_to_view", "arm_to_view"),
    ("learned_router", "router"),
)


def normalize_presets(raw_presets) -> list[tuple[str, str]]:
    raw_presets = (
        DEFAULT_ROUTER_PRESETS if raw_presets is None else raw_presets
    )
    result = []
    names = set()
    for preset in raw_presets:
        if not isinstance(preset, (list, tuple)) or len(preset) != 2:
            raise ValueError("router_presets每项必须是(name, router_mode)。")
        name, mode = str(preset[0]), str(preset[1])
        if Path(name).name != name or not name or name in names:
            raise ValueError(f"Router消融名称非法或重复: {name!r}")
        if mode not in {"none", "arm_to_view", "router"}:
            raise ValueError(f"Router消融模式非法: {mode!r}")
        names.add(name)
        result.append((name, mode))
    return result


def _output_root(checkpoint: Path, eval_cfg) -> Path:
    requested = getattr(eval_cfg, "eval_output_dir", None)
    base = (
        resolve_eval_path(requested)
        if requested
        else checkpoint / "none_av_router_ablation"
    )
    identity = (
        f"eval_seed={int(eval_cfg.seed)}_ep={int(eval_cfg.n_episodes)}"
    )
    return base if base.name == identity else base / identity


def _preset_cfg(eval_cfg, checkpoint: Path, output: Path, mode: str):
    cfg = SimpleNamespace(**vars(eval_cfg))
    cfg.ckpt_path = str(checkpoint)
    cfg.eval_output_dir = str(output)
    cfg.router_mode = mode
    cfg.view_to_arm_output_scale = None
    cfg.arm_to_view_output_scale = None
    cfg.view_to_arm_coupling_scale = None
    cfg.arm_to_view_coupling_scale = None
    cfg.keep_top_after_eval = None
    cfg.prune_checkpoints = False
    cfg.prune_eval_outputs = False
    cfg.allow_prune_with_max_checkpoints = False
    if isinstance(getattr(eval_cfg, "render_camera", None), list):
        cfg.render_camera = list(eval_cfg.render_camera)
    return cfg


def _save_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with open(
        output_root / "none_av_router_ablation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(
        output_root / "none_av_router_ablation_summary.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(eval_cfg) -> list[dict]:
    checkpoint = resolve_eval_path(eval_cfg.ckpt_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Router checkpoint不存在: {checkpoint}")
    output_root = _output_root(checkpoint, eval_cfg)
    presets = normalize_presets(
        getattr(eval_cfg, "router_presets", None)
    )
    rows = []
    for index, (name, mode) in enumerate(presets, start=1):
        logging.info(
            "[%d/%d] Router消融: %s (%s)",
            index,
            len(presets),
            name,
            mode,
        )
        result_rows = evaluate_policy(
            _preset_cfg(
                eval_cfg,
                checkpoint,
                output_root / name,
                mode,
            )
        )
        rows.extend(
            {
                "preset": name,
                "requested_router_mode": mode,
                **row,
            }
            for row in result_rows
        )
        _save_summary(output_root, rows)
    return rows


if __name__ == "__main__":
    init_logging()
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)

    eval_cfg = SimpleNamespace(
        seed=1400,
        ckpt_path=(
            "outputs/2_pretrain/none_av_router/InsertCylinder-3Arms-v0/"
            "REPLACE_WITH_RUN/checkpoints/REPLACE_WITH_CHECKPOINT"
        ),
        checkpoint_source="all",
        max_checkpoints=None,
        eval_output_dir=None,
        continue_on_error=False,
        router_presets=[
            ("none", "none"),
            ("always_arm_to_view", "arm_to_view"),
            ("learned_router", "router"),
        ],
        router_threshold=None,  # None沿用验证集校准后写入checkpoint的阈值。
        mode="strict",
        n_episodes=200,
        max_episodes_rendered=0,
        fps=25,
        max_steps=400,
        batch_size=15,
        use_async_envs=True,
        device="cuda:0",
        deterministic=False,
        render_camera=["zed_cam_left"],
        use_amp=True,
        keep_top_after_eval=None,
        prune_checkpoints=False,
        prune_eval_outputs=False,
        allow_prune_with_max_checkpoints=False,
    )
    if eval_cfg.deterministic or eval_cfg.mode == "strict" or DETERMINISTIC_EVAL:
        ensure_python_hash_seed(eval_cfg.seed)
    main(eval_cfg)
