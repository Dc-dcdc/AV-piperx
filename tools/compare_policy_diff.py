import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config


def resolve_policy_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    nested = path / "pretrained_model"
    if nested.exists():
        return nested
    return path


def find_config_yaml(policy_dir: Path) -> Path:
    candidates = [policy_dir / "config.yaml", policy_dir.parent / "config.yaml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find config.yaml near {policy_dir}")


def load_policy(path: str | Path, device: torch.device):
    policy_dir = resolve_policy_dir(path)
    config_yaml = find_config_yaml(policy_dir)
    hydra_cfg = init_hydra_config(str(config_yaml))
    policy = make_policy(
        hydra_cfg=hydra_cfg,
        pretrained_policy_name_or_path=str(policy_dir),
    )
    policy.to(device)
    policy.eval()
    return policy, policy_dir, config_yaml


def named_tensor_map(module: torch.nn.Module, include_buffers: bool) -> dict[str, torch.Tensor]:
    tensors = {name: value.detach().cpu() for name, value in module.named_parameters()}
    if include_buffers:
        for name, value in module.named_buffers():
            tensors[f"[buffer].{name}"] = value.detach().cpu()
    return tensors


def module_group(name: str) -> str:
    raw = name.removeprefix("[buffer].")
    if "unet" in raw:
        return "diffusion.unet"
    if any(token in raw for token in ("rgb_encoder", "image_encoder", "vision", "backbone", "resnet")):
        return "vision_encoder"
    if "normalizer" in raw or "normalize" in raw:
        return "normalizer"
    if "noise_scheduler" in raw or "scheduler" in raw:
        return "scheduler"
    if raw.startswith("diffusion."):
        return "diffusion.other"
    return raw.split(".")[0]


def empty_aggregate() -> dict:
    return {
        "num_tensors": 0,
        "numel": 0,
        "base_l2_sq": 0.0,
        "diff_l2_sq": 0.0,
        "abs_sum": 0.0,
        "max_abs": 0.0,
    }


def add_to_aggregate(agg: dict, base: torch.Tensor, diff: torch.Tensor) -> None:
    base_f = base.float()
    diff_f = diff.float()
    agg["num_tensors"] += 1
    agg["numel"] += int(diff_f.numel())
    agg["base_l2_sq"] += float(torch.sum(base_f * base_f).item())
    agg["diff_l2_sq"] += float(torch.sum(diff_f * diff_f).item())
    agg["abs_sum"] += float(torch.sum(torch.abs(diff_f)).item())
    if diff_f.numel() > 0:
        agg["max_abs"] = max(agg["max_abs"], float(torch.max(torch.abs(diff_f)).item()))


def finish_aggregate(agg: dict) -> dict:
    base_l2 = math.sqrt(agg["base_l2_sq"])
    diff_l2 = math.sqrt(agg["diff_l2_sq"])
    numel = max(agg["numel"], 1)
    return {
        "num_tensors": agg["num_tensors"],
        "numel": agg["numel"],
        "base_l2": base_l2,
        "diff_l2": diff_l2,
        "relative_l2": diff_l2 / (base_l2 + 1e-12),
        "mean_abs": agg["abs_sum"] / numel,
        "max_abs": agg["max_abs"],
    }


def compare_tensors(base_tensors: dict[str, torch.Tensor], tuned_tensors: dict[str, torch.Tensor], top_k: int) -> dict:
    base_keys = set(base_tensors)
    tuned_keys = set(tuned_tensors)
    common = sorted(base_keys & tuned_keys)
    missing_in_tuned = sorted(base_keys - tuned_keys)
    missing_in_base = sorted(tuned_keys - base_keys)

    rows = []
    global_agg = empty_aggregate()
    groups: dict[str, dict] = {}
    shape_mismatch = []

    for name in common:
        base = base_tensors[name]
        tuned = tuned_tensors[name]
        if tuple(base.shape) != tuple(tuned.shape):
            shape_mismatch.append({"name": name, "base_shape": list(base.shape), "tuned_shape": list(tuned.shape)})
            continue

        diff = tuned.float() - base.float()
        base_l2 = float(torch.linalg.vector_norm(base.float()).item())
        diff_l2 = float(torch.linalg.vector_norm(diff).item())
        abs_diff = torch.abs(diff)
        row = {
            "name": name,
            "group": module_group(name),
            "shape": list(base.shape),
            "numel": int(base.numel()),
            "base_l2": base_l2,
            "diff_l2": diff_l2,
            "relative_l2": diff_l2 / (base_l2 + 1e-12),
            "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
            "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        }
        rows.append(row)
        add_to_aggregate(global_agg, base, diff)
        group = groups.setdefault(row["group"], empty_aggregate())
        add_to_aggregate(group, base, diff)

    rows_by_rel = sorted(rows, key=lambda item: item["relative_l2"], reverse=True)
    rows_by_abs = sorted(rows, key=lambda item: item["max_abs"], reverse=True)
    finished_groups = {name: finish_aggregate(agg) for name, agg in groups.items()}
    groups_by_rel = dict(
        sorted(finished_groups.items(), key=lambda item: item[1]["relative_l2"], reverse=True)
    )

    return {
        "summary": finish_aggregate(global_agg),
        "groups": groups_by_rel,
        "top_relative_l2": rows_by_rel[:top_k],
        "top_max_abs": rows_by_abs[:top_k],
        "missing_in_tuned": missing_in_tuned,
        "missing_in_base": missing_in_base,
        "shape_mismatch": shape_mismatch,
    }


def prepare_obs_for_policy(obs, policy, device: torch.device) -> dict[str, torch.Tensor]:
    def add_batch(obj):
        if isinstance(obj, dict):
            return {key: add_batch(value) for key, value in obj.items()}
        if hasattr(obj, "copy"):
            return np.expand_dims(obj.copy(), axis=0).copy()
        return obj

    batch = preprocess_observation(add_batch(obs))
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in policy.config.input_shapes
    }


def read_env_id(config_yaml: Path) -> tuple[str, str]:
    with open(config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get("env", {}) if isinstance(cfg, dict) else {}
    return env_cfg.get("name", "guided_vision"), env_cfg.get("task", "InsertCylinder-3Arms-v0")


def compare_actions(
    base_policy,
    tuned_policy,
    config_yaml: Path,
    device: torch.device,
    episodes: int,
    max_steps: int,
    seed: int,
    rollout_source: str,
) -> dict:
    import gymnasium as gym
    import env.task.sim_envs  # noqa: F401

    env_name, env_task = read_env_id(config_yaml)
    env_id = f"{env_name}/{env_task}"
    ref_cams = [
        key.replace("observation.images.", "")
        for key in base_policy.config.input_shapes
        if "observation.images." in key
    ]
    env = gym.make(id=env_id, cameras=ref_cams)

    stats = empty_aggregate()
    per_episode = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            if hasattr(base_policy, "reset"):
                base_policy.reset()
            if hasattr(tuned_policy, "reset"):
                tuned_policy.reset()

            ep_stats = empty_aggregate()
            done = False
            step = 0
            while not done and step < max_steps:
                base_obs = prepare_obs_for_policy(obs, base_policy, device)
                tuned_obs = prepare_obs_for_policy(obs, tuned_policy, device)
                action_seed = seed + ep * 100000 + step

                with torch.no_grad():
                    torch.manual_seed(action_seed)
                    np.random.seed(action_seed % (2**32 - 1))
                    random.seed(action_seed)
                    base_action = base_policy.select_action(base_obs)

                    torch.manual_seed(action_seed)
                    np.random.seed(action_seed % (2**32 - 1))
                    random.seed(action_seed)
                    tuned_action = tuned_policy.select_action(tuned_obs)

                diff = tuned_action.detach().cpu().float() - base_action.detach().cpu().float()
                add_to_aggregate(stats, base_action.detach().cpu(), diff)
                add_to_aggregate(ep_stats, base_action.detach().cpu(), diff)

                source_action = tuned_action if rollout_source == "tuned" else base_action
                obs, _, terminated, truncated, _ = env.step(source_action.squeeze(0).detach().cpu().numpy())
                done = bool(terminated or truncated)
                step += 1

            ep_result = finish_aggregate(ep_stats)
            ep_result["steps"] = step
            per_episode.append(ep_result)
    finally:
        env.close()

    return {
        "env_id": env_id,
        "cameras": ref_cams,
        "episodes": episodes,
        "max_steps": max_steps,
        "rollout_source": rollout_source,
        "summary": finish_aggregate(stats),
        "per_episode": per_episode,
    }


def compact_tensor_name(name: str, max_len: int = 92) -> str:
    name = name.removeprefix("[buffer].")
    for prefix in ("diffusion.unet.", "diffusion.rgb_encoder.", "diffusion."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if len(name) <= max_len:
        return name
    return f"{name[:42]}...{name[-42:]}"


def tensor_region(name: str) -> str:
    raw = name.removeprefix("[buffer].")
    if name.startswith("[buffer]."):
        return "Buffer统计量"
    if "diffusion.unet.down_modules" in raw:
        return "UNet下采样"
    if "diffusion.unet.mid_modules" in raw:
        return "UNet中间层"
    if "diffusion.unet.up_modules" in raw:
        return "UNet上采样"
    if "diffusion.unet.final_conv" in raw:
        return "UNet输出层"
    if "diffusion.unet.diffusion_step_encoder" in raw:
        return "UNet时间编码"
    if "rgb_encoder" in raw:
        return "视觉编码器"
    if "normalizer" in raw:
        return "Normalizer"
    return module_group(name)


def print_top_region_summary(rows: list[dict], limit: int) -> None:
    region_stats: dict[str, dict] = {}
    for row in rows[:limit]:
        region = tensor_region(row["name"])
        stats = region_stats.setdefault(region, {"count": 0, "max_rel": 0.0, "sum_rel": 0.0})
        stats["count"] += 1
        stats["max_rel"] = max(stats["max_rel"], row["relative_l2"])
        stats["sum_rel"] += row["relative_l2"]

    if not region_stats:
        return

    print(f"  变化最大张量分布（Top {limit}）:")
    sorted_regions = sorted(region_stats.items(), key=lambda item: item[1]["max_rel"], reverse=True)
    for region, stats in sorted_regions:
        avg_rel = stats["sum_rel"] / max(stats["count"], 1)
        print(
            f"    {region:16s} 数量={stats['count']:2d}, "
            f"最大变化={stats['max_rel'] * 100:.4f}%, 平均变化={avg_rel * 100:.4f}%"
        )


def print_summary(title: str, result: dict, top_k: int, print_top_k: int) -> None:
    summary = result["summary"]
    title_cn = "参数与Buffer差异" if title == "parameter_and_buffer_diff" else "可学习参数差异"
    print(f"\n[{title_cn}]")
    print(f"  整体可学习参数变化: 约 {summary['relative_l2'] * 100:.4f}%")
    group_labels = {
        "diffusion.unet": "Diffusion UNet变化",
        "vision_encoder": "视觉编码器变化",
        "normalizer": "Normalizer变化",
        "scheduler": "Scheduler变化",
        "diffusion.other": "Diffusion其他部分变化",
    }
    for group_name, label in group_labels.items():
        if group_name in result["groups"]:
            rel_percent = result["groups"][group_name]["relative_l2"] * 100
            print(f"  {label}: 约 {rel_percent:.4f}%")

    print("\n  详细数值:")
    print(
        "    张量数量={num_tensors}, 参数量={numel}, 相对L2={relative_l2:.6e}, "
        "差异L2={diff_l2:.6e}, 平均绝对差={mean_abs:.6e}, 最大绝对差={max_abs:.6e}".format(**summary)
    )
    if result["missing_in_tuned"] or result["missing_in_base"] or result["shape_mismatch"]:
        print(
            f"    微调模型缺失={len(result['missing_in_tuned'])}, "
            f"原模型缺失={len(result['missing_in_base'])}, "
            f"形状不匹配={len(result['shape_mismatch'])}"
        )

    print("  按模块统计:")
    for group, stats in list(result["groups"].items())[: min(8, top_k)]:
        print(
            f"    {group:24s} 变化={stats['relative_l2'] * 100:.4f}% "
            f"相对L2={stats['relative_l2']:.6e} "
            f"平均绝对差={stats['mean_abs']:.6e} 最大绝对差={stats['max_abs']:.6e}"
        )

    rows_to_print = result["top_relative_l2"][: min(print_top_k, top_k)]
    if rows_to_print:
        print_top_region_summary(result["top_relative_l2"], len(rows_to_print))

    print(f"  变化最大的张量（仅显示Top {len(rows_to_print)}，完整明细见JSON）:")
    for idx, row in enumerate(rows_to_print, start=1):
        print(
            f"    {idx:02d}. {tensor_region(row['name']):12s} "
            f"变化={row['relative_l2'] * 100:8.4f}% "
            f"最大差={row['max_abs']:.3e} "
            f"{compact_tensor_name(row['name'])}"
        )


def build_runtime_config(args: argparse.Namespace) -> dict:
    cfg = dict(FILE_CONFIG) if USE_FILE_CONFIG else {}

    cli_overrides = {
        "base": args.base,
        "tuned": args.tuned,
        "device": args.device,
        "top_k": args.top_k,
        "print_top_k": args.print_top_k,
        "action_episodes": args.action_episodes,
        "action_max_steps": args.action_max_steps,
        "rollout_source": args.rollout_source,
        "seed": args.seed,
        "output": args.output,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            cfg[key] = value

    if args.no_buffers:
        cfg["include_buffers"] = False

    missing = [key for key in ("base", "tuned") if not str(cfg.get(key, "")).strip()]
    if missing:
        raise ValueError(
            f"缺少必要配置: {missing}。请在 tools/compare_policy_diff.py 顶部 FILE_CONFIG 中填写，"
            "或通过 --base/--tuned 临时传入。"
        )

    cfg.setdefault("device", "cuda:0")
    cfg.setdefault("top_k", 20)
    cfg.setdefault("print_top_k", 10)
    cfg.setdefault("include_buffers", True)
    cfg.setdefault("action_episodes", 0)
    cfg.setdefault("action_max_steps", 300)
    cfg.setdefault("rollout_source", "base")
    cfg.setdefault("seed", 1000)
    cfg.setdefault("output", "")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two LeRobot policy checkpoints.")
    parser.add_argument("--base", default=None, help="Temporarily override FILE_CONFIG['base'].")
    parser.add_argument("--tuned", default=None, help="Temporarily override FILE_CONFIG['tuned'].")
    parser.add_argument("--device", default=None, help="Temporarily override FILE_CONFIG['device'].")
    parser.add_argument("--top-k", type=int, default=None, help="Temporarily override FILE_CONFIG['top_k'].")
    parser.add_argument("--print-top-k", type=int, default=None, help="How many changed tensors to print in terminal.")
    parser.add_argument("--no-buffers", action="store_true", help="Do not compare named buffers.")
    parser.add_argument("--action-episodes", type=int, default=None, help="Temporarily override action episodes.")
    parser.add_argument("--action-max-steps", type=int, default=None, help="Temporarily override action max steps.")
    parser.add_argument("--rollout-source", choices=["base", "tuned"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="Temporarily override report path.")
    args = parser.parse_args()
    cfg = build_runtime_config(args)

    print("Runtime config:")
    for key in (
        "base",
        "tuned",
        "device",
        "top_k",
        "print_top_k",
        "include_buffers",
        "action_episodes",
        "action_max_steps",
        "rollout_source",
        "seed",
        "output",
    ):
        print(f"  {key}: {cfg[key]}")

    device = get_safe_torch_device(cfg["device"], log=True)
    print(f"\nLoading base policy: {cfg['base']}")
    base_policy, base_dir, base_config = load_policy(cfg["base"], device)
    print(f"Loading tuned policy: {cfg['tuned']}")
    tuned_policy, tuned_dir, tuned_config = load_policy(cfg["tuned"], device)

    include_buffers = bool(cfg["include_buffers"])
    base_tensors = named_tensor_map(base_policy, include_buffers=include_buffers)
    tuned_tensors = named_tensor_map(tuned_policy, include_buffers=include_buffers)
    compare_top_k = max(int(cfg["top_k"]), int(cfg["print_top_k"]))
    parameter_result = compare_tensors(base_tensors, tuned_tensors, top_k=compare_top_k)
    print_summary(
        "parameter_and_buffer_diff" if include_buffers else "parameter_diff",
        parameter_result,
        top_k=compare_top_k,
        print_top_k=int(cfg["print_top_k"]),
    )

    report = {
        "base_dir": str(base_dir),
        "tuned_dir": str(tuned_dir),
        "base_config": str(base_config),
        "tuned_config": str(tuned_config),
        "include_buffers": include_buffers,
        "parameter_diff": parameter_result,
    }

    if int(cfg["action_episodes"]) > 0:
        print(f"\nRunning action-output diff for {cfg['action_episodes']} episode(s)...")
        action_result = compare_actions(
            base_policy=base_policy,
            tuned_policy=tuned_policy,
            config_yaml=base_config,
            device=device,
            episodes=int(cfg["action_episodes"]),
            max_steps=int(cfg["action_max_steps"]),
            seed=int(cfg["seed"]),
            rollout_source=str(cfg["rollout_source"]),
        )
        action_summary = action_result["summary"]
        print(
            "[动作输出差异]\n"
            "  环境={env_id}, 回合数={episodes}, 环境推进策略={rollout_source}\n"
            "  动作整体变化: 约 {relative_percent:.4f}%\n"
            "  相对L2={relative_l2:.6e}, 差异L2={diff_l2:.6e}, "
            "平均绝对差={mean_abs:.6e}, 最大绝对差={max_abs:.6e}".format(
                env_id=action_result["env_id"],
                episodes=action_result["episodes"],
                rollout_source=action_result["rollout_source"],
                relative_percent=action_summary["relative_l2"] * 100,
                **action_summary,
            )
        )
        report["action_diff"] = action_result

    if cfg["output"]:
        output_path = Path(cfg["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report to: {output_path}")


# =========================
# 用户配置区：平时只改这里
# =========================
USE_FILE_CONFIG = True

FILE_CONFIG = {
    # 原始预训练模型路径。可以填 checkpoint 目录，也可以直接填 pretrained_model 目录。
    "base": "outputs/1_hugging_model/pre_sim_insert_cylinder_3arms_zed_wrist_diffusion",

    # 微调后模型路径。把这里改成你想检查的第几十轮 checkpoint。
    "tuned": "outputs/3_finetune/train/2026-05-29/19-41-54_InsertCylinder-3Arms-v0_ft_zed_wrist_diffusion/checkpoints/000220_sr=0.80_reward=495.12_Ploss=-0.0024_Vloss=0.5348",

    # 基础对比配置。
    "device": "cuda:0",
    "top_k": 30,        # 保存到 JSON 的变化张量数量。
    "print_top_k": 10,  # 终端只显示前几个，避免输出太乱。
    "include_buffers": False,
    "output": "outputs/model_diff/compare_policy_diff.json",

    # 动作输出差异验证。只想快速看参数差异时保持 0；想启动环境比较动作时改成 3/5/10。
    "action_episodes": 0,
    "action_max_steps": 300,
    "rollout_source": "base",  # base 或 tuned；表示动作差异验证时用哪个模型的动作推进环境。
    "seed": 1000,
}


if __name__ == "__main__":
    main()
