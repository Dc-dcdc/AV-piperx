#!/usr/bin/env python3
"""Visualize depth from the MuJoCo ZED cameras.

This utility is for the simulated ZED cameras defined in the MuJoCo XML, not a
real ZED camera. It reads `zed_cam_left` and `zed_cam_right` images from
dm_control/MuJoCo, then visualizes:

1. MuJoCo renderer depth: ground-truth camera depth from `physics.render(..., depth=True)`.
2. Stereo depth: estimated from the left/right ZED images with OpenCV SGBM when available,
   otherwise with a pure NumPy block-matching fallback.

Controls:
    q / Esc : quit
    s       : save the current RGB/depth/disparity outputs
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

# 这个脚本主要做 MuJoCo 相机离屏渲染。若继承到外部的 MUJOCO_GL=glfw，
# 在部分桌面/远程显示环境会触发 GLXBadDrawable / X_GLXMakeCurrent。
# 默认强制使用 EGL；确实需要改后端时，用 SIM_ZED_MUJOCO_GL=osmesa/glfw 覆盖。
os.environ["MUJOCO_GL"] = os.environ.get("SIM_ZED_MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

try:
    import cv2
except ImportError:
    cv2 = None

import imageio.v2 as imageio
import numpy as np
from dm_control import mjcf
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = PROJECT_ROOT / "env" / "assets" / "task_insert_cylinder_piper.xml"


def is_cv2_usable() -> bool:
    if cv2 is None:
        return False
    try:
        test_img = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    except Exception:
        return False
    return True


CV2_USABLE = is_cv2_usable()


@dataclass
class SimZedConfig:
    xml_path: Path = DEFAULT_XML_PATH
    left_camera: str = "zed_cam_left"
    right_camera: str = "zed_cam_right"
    width: int = 640
    height: int = 480
    depth_min_m: float = 0.10
    depth_max_m: float = 1.50
    zed_baseline_m: float = 0.06
    zed_fovy_deg: float = 66.21
    stereo_num_disparities: int = 128
    stereo_block_size: int = 7
    step_sim: bool = False
    sim_steps_per_frame: int = 1
    output_dir: Path = PROJECT_ROOT / "outputs" / "zed_depth"
    save_on_start: bool = False
    display: bool = False


def load_physics(xml_path: Path):
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    mjcf_root = mjcf.from_path(str(xml_path))
    physics = mjcf.Physics.from_mjcf_model(mjcf_root)
    physics.forward()
    return physics


def render_rgb_and_depth(physics, camera: str, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.ascontiguousarray(physics.render(height=height, width=width, camera_id=camera))
    depth = np.ascontiguousarray(physics.render(height=height, width=width, camera_id=camera, depth=True))
    return rgb, depth.astype(np.float32, copy=False)


def focal_pixels_from_vertical_fovy(height: int, fovy_deg: float) -> float:
    fovy_rad = math.radians(fovy_deg)
    return height / (2.0 * math.tan(fovy_rad / 2.0))


def make_stereo_matcher(num_disparities: int, block_size: int):
    if not CV2_USABLE:
        return None

    num_disparities = max(16, int(math.ceil(num_disparities / 16)) * 16)
    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def rgb_to_gray_float(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def box_filter_mean(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    kernel = 2 * radius + 1
    total = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    )
    return total / float(kernel * kernel)


def estimate_stereo_depth_numpy(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    focal_px: float,
    baseline_m: float,
    num_disparities: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy stereo block matching fallback.

    It uses only the left/right RGB images to estimate disparity, then converts it
    to metric depth with depth = focal * baseline / disparity.
    """
    left = rgb_to_gray_float(left_rgb)
    right = rgb_to_gray_float(right_rgb)
    h, w = left.shape
    min_disparity = 1
    max_disparity = min(max(16, int(num_disparities)), w - 1)
    block_radius = max(int(block_size) // 2, 1)

    best_cost = np.full((h, w), np.inf, dtype=np.float32)
    best_disp = np.zeros((h, w), dtype=np.float32)
    for disp in range(min_disparity, max_disparity + 1):
        shifted_right = np.empty_like(right)
        shifted_right[:, :disp] = right[:, :1]
        shifted_right[:, disp:] = right[:, :-disp]
        cost = np.abs(left - shifted_right)
        cost[:, :disp] = 1e3
        cost = box_filter_mean(cost, block_radius)

        update = cost < best_cost
        best_cost[update] = cost[update]
        best_disp[update] = float(disp)

    texture = box_filter_mean(np.abs(left - box_filter_mean(left, block_radius)), block_radius)
    valid = np.isfinite(best_cost) & (best_cost < 0.18) & (texture > 0.01)
    best_disp[~valid] = 0.0

    depth = np.full(best_disp.shape, np.nan, dtype=np.float32)
    valid = best_disp > 0.1
    depth[valid] = (float(focal_px) * float(baseline_m)) / best_disp[valid]
    return depth, best_disp


def estimate_stereo_depth(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    matcher,
    focal_px: float,
    baseline_m: float,
    num_disparities: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if matcher is None:
        return estimate_stereo_depth_numpy(
            left_rgb,
            right_rgb,
            focal_px=focal_px,
            baseline_m=baseline_m,
            num_disparities=num_disparities,
            block_size=block_size,
        )

    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

    valid = disparity > 0.1
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = (focal_px * baseline_m) / disparity[valid]
    return depth, disparity


def sanitize_depth(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    depth = depth_m.astype(np.float32, copy=True)
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    depth[invalid] = np.nan
    return np.clip(depth, min_m, max_m)


def depth_to_colormap(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    depth = sanitize_depth(depth_m, min_m, max_m)
    normalized = (depth - min_m) / max(max_m - min_m, 1e-6)
    normalized = np.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=1.0)
    heat = 1.0 - normalized
    colored = scalar_to_rgb(heat)
    colored[np.isnan(depth)] = (0, 0, 0)
    return colored


def disparity_to_colormap(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 0.1
    normalized = np.zeros(disparity.shape, dtype=np.float32)
    if np.any(valid):
        vmax = np.percentile(disparity[valid], 99)
        normalized = np.clip(disparity / max(vmax, 1e-6), 0, 1)
    colored = scalar_to_rgb(normalized)
    colored[~valid] = (0, 0, 0)
    return colored


def scalar_to_rgb(value: np.ndarray) -> np.ndarray:
    """Map a scalar image in [0, 1] to a simple blue-green-red heatmap."""
    x = np.clip(value.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def draw_depth_hud(image_rgb: np.ndarray, depth_m: np.ndarray, label: str) -> np.ndarray:
    vis = image_rgb.copy()
    h, w = depth_m.shape[:2]
    cx, cy = w // 2, h // 2
    patch = depth_m[max(0, cy - 3) : min(h, cy + 4), max(0, cx - 3) : min(w, cx + 4)]
    valid_patch = patch[np.isfinite(patch) & (patch > 0.0)]
    center_depth = float(np.nanmedian(valid_patch)) if valid_patch.size else float("nan")
    valid_ratio = float(np.mean(np.isfinite(depth_m) & (depth_m > 0.0)))

    image = Image.fromarray(vis)
    draw = ImageDraw.Draw(image)
    draw.line((cx - 11, cy, cx + 11, cy), fill=(255, 255, 0), width=2)
    draw.line((cx, cy - 11, cx, cy + 11), fill=(255, 255, 0), width=2)
    text = f"{label}: center={center_depth:.3f} m | valid={valid_ratio * 100:.1f}%"
    draw.text((13, 13), text, fill=(20, 20, 20))
    draw.text((12, 12), text, fill=(255, 255, 255))
    return np.asarray(image)


def save_frame(
    output_dir: Path,
    index: int,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    mujoco_depth_m: np.ndarray,
    stereo_depth_m: np.ndarray,
    disparity: np.ndarray,
    mujoco_depth_color: np.ndarray,
    stereo_depth_color: np.ndarray,
    preview: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sim_zed_{index:06d}"

    imageio.imwrite(output_dir / f"{stem}_left.png", left_rgb)
    imageio.imwrite(output_dir / f"{stem}_right.png", right_rgb)
    imageio.imwrite(output_dir / f"{stem}_mujoco_depth_color.png", mujoco_depth_color)
    imageio.imwrite(output_dir / f"{stem}_stereo_depth_color.png", stereo_depth_color)
    if preview is not None:
        imageio.imwrite(output_dir / f"{stem}_preview.png", preview)
    np.save(output_dir / f"{stem}_mujoco_depth_m.npy", mujoco_depth_m.astype(np.float32))
    np.save(output_dir / f"{stem}_stereo_depth_m.npy", stereo_depth_m.astype(np.float32))
    np.save(output_dir / f"{stem}_stereo_disparity_px.npy", disparity.astype(np.float32))
    print(f"Saved frame group: {output_dir / stem}")


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR))


def draw_label(image: np.ndarray, text: str, xy: tuple[int, int] = (12, 12)) -> np.ndarray:
    pil_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_image)
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(20, 20, 20))
    draw.text((x, y), text, fill=(255, 255, 255))
    return np.asarray(pil_image)


def build_preview(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    mujoco_depth_m: np.ndarray,
    stereo_depth_m: np.ndarray,
    disparity: np.ndarray,
    cfg: SimZedConfig,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mujoco_depth_color = depth_to_colormap(mujoco_depth_m, cfg.depth_min_m, cfg.depth_max_m)
    stereo_depth_color = depth_to_colormap(stereo_depth_m, cfg.depth_min_m, cfg.depth_max_m)
    disparity_color = disparity_to_colormap(disparity)

    left_hud = draw_depth_hud(left_rgb, mujoco_depth_m, "MuJoCo depth")
    stereo_hud = draw_depth_hud(stereo_depth_color, stereo_depth_m, "Stereo depth")
    stereo_pair = np.hstack((left_rgb, right_rgb))
    stereo_pair = resize_rgb(stereo_pair, (cfg.width, cfg.height))
    stereo_pair = draw_label(stereo_pair, "Left / Right RGB")
    disparity_color = draw_label(disparity_color, "Stereo disparity")

    overlay = (left_rgb.astype(np.float32) * 0.55 + mujoco_depth_color.astype(np.float32) * 0.45).astype(np.uint8)
    overlay = draw_label(overlay, f"MuJoCo overlay | FPS {fps:.1f}")

    top = np.hstack((left_hud, mujoco_depth_color, overlay))
    bottom = np.hstack((stereo_pair, stereo_hud, disparity_color))
    preview = np.vstack((top, bottom))
    return preview, mujoco_depth_color, stereo_depth_color


def run(cfg: SimZedConfig) -> None:
    physics = load_physics(cfg.xml_path)
    matcher = make_stereo_matcher(cfg.stereo_num_disparities, cfg.stereo_block_size)
    focal_px = focal_pixels_from_vertical_fovy(cfg.height, cfg.zed_fovy_deg)
    display_enabled = cfg.display and CV2_USABLE

    print(f"Loaded MuJoCo XML: {cfg.xml_path}")
    print(f"MuJoCo GL backend: {os.environ.get('MUJOCO_GL')}")
    print(f"Rendering cameras: {cfg.left_camera}, {cfg.right_camera}")
    print(f"Stereo depth uses baseline={cfg.zed_baseline_m:.3f} m, focal={focal_px:.1f} px")
    if matcher is None:
        print("OpenCV SGBM is unavailable; using pure NumPy stereo block matching.")
    if display_enabled:
        print("Press 's' to save, 'q' or Esc to quit.")
    else:
        print("Live display is disabled; one frame will be saved automatically.")

    frame_index = 0
    save_index = 0
    last_time = time.perf_counter()

    while True:
        if cfg.step_sim:
            physics.step(nstep=cfg.sim_steps_per_frame)

        left_rgb, mujoco_depth_m = render_rgb_and_depth(physics, cfg.left_camera, cfg.height, cfg.width)
        right_rgb, _ = render_rgb_and_depth(physics, cfg.right_camera, cfg.height, cfg.width)
        stereo_depth_m, disparity = estimate_stereo_depth(
            left_rgb,
            right_rgb,
        matcher,
        focal_px=focal_px,
        baseline_m=cfg.zed_baseline_m,
        num_disparities=cfg.stereo_num_disparities,
        block_size=cfg.stereo_block_size,
    )

        now = time.perf_counter()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        preview, mujoco_depth_color, stereo_depth_color = build_preview(
            left_rgb,
            right_rgb,
            mujoco_depth_m,
            stereo_depth_m,
            disparity,
            cfg,
            fps,
        )

        if (cfg.save_on_start or not display_enabled) and frame_index == 0:
            save_frame(
                cfg.output_dir,
                save_index,
                left_rgb,
                right_rgb,
                mujoco_depth_m,
                stereo_depth_m,
                disparity,
                mujoco_depth_color,
                stereo_depth_color,
                preview,
            )
            save_index += 1

        if not display_enabled:
            break

        cv2.imshow("MuJoCo ZED Depth: RGB / Ground Truth / Stereo", preview[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            save_frame(
                cfg.output_dir,
                save_index,
                left_rgb,
                right_rgb,
                mujoco_depth_m,
                stereo_depth_m,
                disparity,
                mujoco_depth_color,
                stereo_depth_color,
                preview,
            )
            save_index += 1

        frame_index += 1

    if CV2_USABLE:
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and visualize depth from MuJoCo simulated ZED cameras.")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML_PATH, help="MuJoCo XML path.")
    parser.add_argument("--left-camera", default="zed_cam_left", help="Left ZED camera name in the XML.")
    parser.add_argument("--right-camera", default="zed_cam_right", help="Right ZED camera name in the XML.")
    parser.add_argument("--width", type=int, default=640, help="Rendered image width.")
    parser.add_argument("--height", type=int, default=480, help="Rendered image height.")
    parser.add_argument("--depth-min-m", type=float, default=0.10, help="Minimum depth for visualization.")
    parser.add_argument("--depth-max-m", type=float, default=1.50, help="Maximum depth for visualization.")
    parser.add_argument("--zed-baseline-m", type=float, default=0.06, help="ZED left/right baseline in meters.")
    parser.add_argument("--zed-fovy-deg", type=float, default=66.21, help="Vertical FOV of the simulated ZED camera.")
    parser.add_argument("--stereo-num-disparities", type=int, default=128, help="Stereo disparity search range.")
    parser.add_argument("--stereo-block-size", type=int, default=7, help="Stereo block matching window size, must be odd.")
    parser.add_argument("--step-sim", action="store_true", help="Step the MuJoCo simulation while visualizing.")
    parser.add_argument("--sim-steps-per-frame", type=int, default=1, help="MuJoCo steps per displayed frame.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "zed_depth")
    parser.add_argument("--save-on-start", action="store_true", help="Save the first rendered frame group.")
    parser.add_argument("--display", action="store_true", help="Open a live OpenCV window. Disabled by default to avoid GLX conflicts.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = SimZedConfig(
        xml_path=args.xml_path,
        left_camera=args.left_camera,
        right_camera=args.right_camera,
        width=args.width,
        height=args.height,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        zed_baseline_m=args.zed_baseline_m,
        zed_fovy_deg=args.zed_fovy_deg,
        stereo_num_disparities=args.stereo_num_disparities,
        stereo_block_size=args.stereo_block_size,
        step_sim=args.step_sim,
        sim_steps_per_frame=args.sim_steps_per_frame,
        output_dir=args.output_dir,
        save_on_start=args.save_on_start,
        display=args.display,
    )
    run(cfg)


if __name__ == "__main__":
    main()
