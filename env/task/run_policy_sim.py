from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import yaml
from lerobot.common.envs.utils import preprocess_observation

import env as _registered_env  # noqa: F401  Register Gym environments.
from env.constants import SIM_DT


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class RunPolicySimConfig:
    ckpt_path: str
    display_cameras: tuple[str, ...] = (
        "zed_cam_left",
        "zed_cam_right",
        "wrist_cam_left",
        "wrist_cam_right",
        "overhead_cam",
        "worms_eye_cam",
    )
    zed_left_camera: str = "zed_cam_left"
    zed_right_camera: str = "zed_cam_right"
    show_zed_depth: bool = True
    show_mujoco_viewer: bool = True
    device: str = "cuda"
    image_height: int = 480
    image_width: int = 640
    window_name: str = "Multi-Camera Monitor"
    window_width: int = 1280
    window_height: int = 960
    max_grid_cols: int = 2
    depth_min_m: float = 0.10
    depth_max_m: float = 1.50
    stereo_num_disparities: int = 128
    stereo_block_size: int = 7
    control_dt: float = SIM_DT


@dataclass
class ControlState:
    run_policy: bool
    force_reset: bool = False
    quit: bool = False


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def resolve_checkpoint_dirs(ckpt_path: str | Path) -> tuple[Path, Path, Path]:
    ckpt_root = resolve_path(ckpt_path)
    if not ckpt_root.exists():
        raise FileNotFoundError(
            f"找不到权重路径: {ckpt_root}\n"
            "请在文件最下面的 CONFIG.ckpt_path 中改成正确路径。"
        )

    if (ckpt_root / "pretrained_model").is_dir():
        load_dir = ckpt_root / "pretrained_model"
        root_dir = ckpt_root
    elif ckpt_root.name == "pretrained_model":
        load_dir = ckpt_root
        root_dir = ckpt_root.parent
    else:
        load_dir = ckpt_root
        root_dir = ckpt_root

    config_yaml_path = load_dir / "config.yaml"
    if not config_yaml_path.exists():
        config_yaml_path = root_dir / "config.yaml"
    if not config_yaml_path.exists():
        raise FileNotFoundError(
            f"找不到模型配置文件 config.yaml: {load_dir} 或 {root_dir}"
        )

    return root_dir, load_dir, config_yaml_path

# 加载模型权重、环境配置等，并创建环境实例
def load_policy(cfg: RunPolicySimConfig):
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.utils.utils import init_hydra_config

    root_dir, load_dir, config_yaml_path = resolve_checkpoint_dirs(cfg.ckpt_path)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    print(f"初始化推理程序: device={device}")
    print(f"加载策略权重: {load_dir}")

    hydra_cfg = init_hydra_config(str(config_yaml_path))
    policy = make_policy(
        hydra_cfg=hydra_cfg,
        pretrained_policy_name_or_path=str(load_dir),
    )
    policy.to(device)
    policy.eval()

    with open(config_yaml_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}

    return policy, full_cfg, device, root_dir, load_dir


def make_env(policy, full_cfg: dict, cfg: RunPolicySimConfig):
    input_keys = policy.config.input_shapes.keys()
    ref_cams = [
        key.replace("observation.images.", "")
        for key in input_keys
        if key.startswith("observation.images.")
    ]
    if not ref_cams:
        raise ValueError("模型配置中未找到 observation.images.* 输入，请检查 checkpoint/config.yaml。")

    obs_cameras = list(
        dict.fromkeys(ref_cams + list(cfg.display_cameras) + [cfg.zed_left_camera, cfg.zed_right_camera])
    )
    env_cfg = full_cfg.get("env", {})
    env_name = env_cfg.get("name", "guided_vision")
    env_task = env_cfg.get("task", "SewNeedle-3Arms-v0")
    env_id = f"{env_name}/{env_task}"

    print(f"初始化环境: {env_id}")
    print(f"策略观测相机: {ref_cams}")
    print(f"环境启用相机: {obs_cameras}")

    env = gym.make(
        id=env_id,
        disable_env_checker=True,
        cameras=obs_cameras,
    )
    return env, ref_cams, obs_cameras


def prepare_obs_for_policy(obs: dict, policy, device) -> dict[str, torch.Tensor]:
    def add_batch_dim(obj):
        if isinstance(obj, dict):
            return {key: add_batch_dim(value) for key, value in obj.items()}
        if hasattr(obj, "copy"):
            return np.expand_dims(obj.copy(), axis=0).copy()
        return obj

    batch = preprocess_observation(add_batch_dim(obs))
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in policy.config.input_shapes
    }


def focal_pixels_from_vertical_fovy(height: int, fovy_deg: float) -> float:
    fovy_rad = math.radians(float(fovy_deg))
    return height / (2.0 * math.tan(fovy_rad / 2.0))


def get_zed_intrinsics_from_sim(sim_env, cfg: RunPolicySimConfig) -> tuple[float, float, float]:
    baseline_m = 0.06
    fovy_deg = 66.21
    try:
        physics = sim_env._physics
        left_id = physics.model.name2id(cfg.zed_left_camera, "camera")
        right_id = physics.model.name2id(cfg.zed_right_camera, "camera")
        baseline_m = float(np.linalg.norm(physics.data.cam_xpos[left_id] - physics.data.cam_xpos[right_id]))
        fovy_deg = float(physics.model.cam_fovy[left_id])
    except Exception:
        pass

    focal_px = focal_pixels_from_vertical_fovy(cfg.image_height, fovy_deg)
    return baseline_m, fovy_deg, focal_px

# 创建一个 OpenCV 的 StereoSGBM 双目立体匹配器，用于从左右相机图像计算 视差图 disparity map。
def make_stereo_matcher(cfg: RunPolicySimConfig):
    num_disparities = max(16, int(math.ceil(cfg.stereo_num_disparities / 16)) * 16)
    
    # 视差越大，表示物体越近；numDisparities 越大，算法能搜索的深度范围越宽，但计算更慢，也更容易引入误匹配。
    block_size = int(cfg.stereo_block_size) # SGBM 的 blockSize 必须是奇数且 >=1
    
    if block_size % 2 == 0:
        block_size += 1
    
    # 窗口越小，细节保留越好，但噪声更大；窗口越大，视差更平滑，但边缘和细小结构可能被抹掉
    block_size = max(3, block_size)

    return cv2.StereoSGBM_create(
        minDisparity=0,                       # 最小视差，通常设为0，表示不允许负视差（即右图像中的点不能比左图像更靠右）
        numDisparities=num_disparities,       # 视差数量，必须是16的倍数，增加可以提升远距离深度分辨率但会增加计算量
        blockSize=block_size,                 # 匹配块大小，必须是奇数，较小的值可以保留更多细节但可能更噪声，较大的值会更平滑但可能丢失边缘信息
        P1=8 * 3 * block_size * block_size,   # 惩罚视差变化较小的像素块，通常设为8*通道数*blockSize^2
        P2=32 * 3 * block_size * block_size,  # 惩罚视差变化较大的像素块，通常设为32*通道数*blockSize^2，P2应该大于P1
        disp12MaxDiff=1,                      # 左右视差图的一致性检查，允许的最大差异，设为1可以过滤掉一些错误匹配
        uniquenessRatio=8,                    # 视差唯一性百分比，较高的值可以减少错误匹配，但可能丢失一些正确匹配，通常设为5-15
        speckleWindowSize=80,                 # 视差连通区域的最大尺寸，较小的值可以过滤掉更多噪声，但可能丢失一些小物体，通常设为50-200
        speckleRange=2,                       # 视差连通区域内允许的最大视差变化，较小的值可以过滤掉更多噪声，但可能丢失一些正确匹配，通常设为1-2
        preFilterCap=63,                      # 预滤波器的截断值，较高的值可以保留更多细节，但可能更噪声，通常设为31-63
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # 使用 SGBM 的三路算法，可以获得更好的匹配质量，但计算更慢，适合离线处理或性能较好的环境
    )


# 从左右相机图像计算深度图，并将深度图转换为可视化的 BGR 图像，同时在中心区域显示当前的深度值和有效像素比例。
def estimate_zed_stereo_depth(left_rgb, right_rgb, stereo_matcher, focal_px: float, baseline_m: float):
    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)

    # 计算视差：disparity = x_left - x_right，单位是像素。
    # 同一个三维点在左图和右图中的横向像素差越大，说明它离相机越近；视差越小，说明它越远。
    disparity_px = stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

    depth_m = np.full(disparity_px.shape, np.nan, dtype=np.float32)
    valid = disparity_px > 0.1 # 过滤掉视差过小的像素，避免除零或过大深度值

    # 根据双目立体视觉的基本原理，深度 = (焦距 * 基线) / 视差  
    depth_m[valid] = (float(focal_px) * float(baseline_m)) / disparity_px[valid]  # Z = fx * B / d
    return depth_m, disparity_px  # 返回深度图和视差图

# 将深度图转换为可视化的 BGR 图像，同时返回一个有效像素的布尔掩码，用于后续统计和显示。
def depth_to_bgr(depth_m: np.ndarray, cfg: RunPolicySimConfig):
    valid = np.isfinite(depth_m) & (depth_m > 0.0)                # 只有有限且正的深度值才是有效的
    clipped = np.clip(depth_m, cfg.depth_min_m, cfg.depth_max_m)  # 将深度值限制在指定范围内，避免极端值影响可视化效果
    normalized = (clipped - cfg.depth_min_m) / max(cfg.depth_max_m - cfg.depth_min_m, 1e-6) # 归一化到 [0, 1] 范围，depth_min_m 映射为 0，depth_max_m 映射为 1
    normalized = np.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=1.0)  # 将无效值（NaN、正无穷、负无穷）替换为 1.0，这样它们在可视化中会显示为最远的颜色（通常是黑色或深蓝色），而不是引入异常的颜色。
    heat = ((1.0 - normalized) * 255.0).astype(np.uint8)          # 将归一化深度映射到 0~255 的整数灰度
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)   
    depth_bgr = cv2.applyColorMap(heat, colormap)                 # 将灰度图变成 BGR 彩色图 
    depth_bgr[~valid] = 0                                         # 将无效像素的颜色设置为黑色（0, 0, 0），以便在显示时区分有效和无效区域
    return depth_bgr, valid


def make_zed_depth_panel(
    left_rgb,
    right_rgb,
    stereo_matcher,
    cfg: RunPolicySimConfig,
    focal_px: float,
    baseline_m: float,
):
    # 计算 ZED 双目立体深度图和视差图
    depth_m, disparity_px = estimate_zed_stereo_depth(
        left_rgb,
        right_rgb,
        stereo_matcher,
        focal_px=focal_px,
        baseline_m=baseline_m,
    )

    # 将深度图转换为可视化的 BGR 图像，同时返回一个有效像素的布尔掩码，用于后续统计和显示。
    depth_bgr, valid = depth_to_bgr(depth_m, cfg) 

    h, w = depth_m.shape[:2]
    cx, cy = w // 2, h // 2   # 获取中心点坐标
    patch = depth_m[max(0, cy - 4):min(h, cy + 5), max(0, cx - 4):min(w, cx + 5)] # 提取中心 9x9 区域的深度值，用于计算中心深度和有效像素比例
    patch = patch[np.isfinite(patch) & (patch > 0.0)]                             # 过滤掉无效的深度值（非有限值或非正值），只保留有效的深度值用于统计
    center_depth = float(np.median(patch)) if patch.size else float("nan")        # 计计算中心区域深度的中值，可以减少噪声的影响，如果没有有效像素则返回 NaN
    valid_ratio = float(np.mean(valid)) if valid.size else 0.0                    # 计算整个深度图中有效像素的比例，作为当前视差计算质量的一个指标

    cv2.line(depth_bgr, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 2)
    cv2.line(depth_bgr, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 2)
    cv2.putText(depth_bgr, "zed_stereo_depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(
        depth_bgr, 
        f"center: {center_depth:.3f} m | valid: {valid_ratio * 100:.1f}%",# 在深度图上显示中心点深度和有效像素百分比
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        depth_bgr,
        f"range: {cfg.depth_min_m:.2f}-{cfg.depth_max_m:.2f} m | baseline: {baseline_m:.3f} m",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    return depth_bgr, depth_m, disparity_px


def label_frame(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    frame = frame_bgr.copy()
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame


def render_camera_rgb(sim_env, camera: str, cfg: RunPolicySimConfig):
    return sim_env._physics.render(
        height=cfg.image_height,
        width=cfg.image_width,
        camera_id=camera,
    )


def build_image_grid(frames_bgr: list[np.ndarray], max_cols: int) -> np.ndarray:
    if not frames_bgr:
        raise ValueError("没有可显示的相机画面。")

    max_cols = max(1, int(max_cols))
    rows = []
    for i in range(0, len(frames_bgr), max_cols):
        row = frames_bgr[i:i + max_cols]
        while len(row) < max_cols:
            row.append(np.zeros_like(frames_bgr[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def render_monitor_frame(
    sim_env,
    cfg: RunPolicySimConfig,
    state: ControlState,
    steps: int,
    episode_reward: float,
    stereo_matcher,          # OpenCV 立体匹配器实例，用于计算 ZED 深度图
    zed_focal_px: float,     # ZED 相机的焦距（单位：像素），用于将视差转换为深度，通过相机参数计算得到
    zed_baseline_m: float,   # ZED 相机的基线距离（单位：米），用于将视差转换为深度，通过相机参数计算得到
) -> np.ndarray:
    frames_bgr = []          # 保存每个相机渲染后的 BGR 图像
    rgb_cache = {}           # 用于缓存已经渲染过的相机图像

    for camera in cfg.display_cameras:
        # 渲染指定相机的 RGB 图像，返回 (H,W,3)
        rgb = render_camera_rgb(sim_env, camera, cfg)
        rgb_cache[camera] = rgb
        # 将 RGB 图像转换为 BGR 并添加标签后保存到 frames_bgr 列表中
        frames_bgr.append(label_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), camera))

    if cfg.show_zed_depth:
        left_rgb = rgb_cache.get(cfg.zed_left_camera)
        right_rgb = rgb_cache.get(cfg.zed_right_camera)
        if left_rgb is None:
            left_rgb = render_camera_rgb(sim_env, cfg.zed_left_camera, cfg)
        if right_rgb is None:
            right_rgb = render_camera_rgb(sim_env, cfg.zed_right_camera, cfg)

        zed_depth_bgr, _, _ = make_zed_depth_panel(
            left_rgb,
            right_rgb,
            stereo_matcher,
            cfg,
            focal_px=zed_focal_px,
            baseline_m=zed_baseline_m,
        )
        frames_bgr.append(zed_depth_bgr)

    # 根据指定列数把多张图像拼接成网格
    combined = build_image_grid(frames_bgr, cfg.max_grid_cols)
    h, w = combined.shape[:2]
    # 左下角显示当前步数和累计奖励
    cv2.putText(
        combined,
        f"Step: {steps} | Reward: {episode_reward:.2f}",
        (800, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    if not state.run_policy:
        text = "PAUSED - Press 'P' to Start"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(combined, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return combined


def launch_mujoco_viewer(sim_env, state: ControlState, enabled: bool):
    if not enabled:
        return None

    import mujoco.viewer

    def key_callback(keycode):
        if keycode == 32:
            # 空格键触发环境重置
            state.force_reset = True
        elif keycode in (ord("p"), ord("P")):
            state.run_policy = True
            print("[键盘指令] 开始策略推理...")
        elif keycode in (ord("q"), ord("Q"), 27):
            # Q 键或 Esc 键触发退出
            state.quit = True

    return mujoco.viewer.launch_passive(
        sim_env._physics.model.ptr,      # 指向内部 mjModel 结构 告诉 Viewer 渲染哪个物理模型
        sim_env._physics.data.ptr,       # 物理状态数据的指针，供 viewer 在渲染时访问物理状态数据（关节位置、速度、碰撞状态等）。
        show_left_ui=True,               # 显示左侧 UI 面板
        show_right_ui=True,              # 显示右侧 UI 面板
        key_callback=key_callback,       # 注册键盘回调函数，监听键盘输入
    )


def handle_cv_key(key: int, state: ControlState):
    if key in (ord(" "),):
        state.force_reset = True
    elif key in (ord("p"), ord("P")):
        state.run_policy = True
        print("[监控窗口指令] 开始策略推理...")
    elif key in (ord("q"), ord("Q"), 27):
        state.quit = True


def run_policy_step(env, policy, obs, device):
    batch = prepare_obs_for_policy(obs, policy, device)
    with torch.no_grad():
        action_tensor = policy.select_action(batch)
    action = action_tensor.squeeze(0).cpu().numpy()
    return env.step(action)


def print_controls():
    print("\n" + "=" * 60)
    print("交互控制说明:")
    print("P       : 开始策略推理")
    print("Space   : 重置环境")
    print("Q / Esc : 退出")
    print("=" * 60 + "\n")


def run(cfg: RunPolicySimConfig):
    # 加载
    policy, full_cfg, device, _, _ = load_policy(cfg)  # 加载模型权重、环境配置等
    env, _, _ = make_env(policy, full_cfg, cfg)        # 创建环境实例，启用指定相机
    sim_env = env.unwrapped                            # 去掉所有环境包装（Wrappers），直接获取底层原始环境实例

    obs, _ = env.reset()
    if hasattr(policy, "reset"): 
        policy.reset()

    state = ControlState(run_policy=False) # 默认不运行策略，等待用户按 P 键开始
    viewer = launch_mujoco_viewer(sim_env, state, cfg.show_mujoco_viewer)  # 启动 Mujoco 视角（如果启用）
    
    # 获取 ZED 相机参数（基线、垂直视场角、焦距像素），用于后续深度计算
    zed_baseline_m, zed_fovy_deg, zed_focal_px = get_zed_intrinsics_from_sim(sim_env, cfg) 
    stereo_matcher = make_stereo_matcher(cfg) # 创建 OpenCV SGBM 立体匹配器，用于计算 ZED 深度图
    print(
        f"ZED 立体深度参数: baseline={zed_baseline_m:.3f} m, "
        f"fovy={zed_fovy_deg:.2f} deg, focal={zed_focal_px:.1f} px"
    )

    print_controls()
    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_name, cfg.window_width, cfg.window_height)

    episode_reward = 0.0
    steps = 0
    render_error_printed = False

    try:
        while not state.quit and (viewer is None or viewer.is_running()):
            step_start = time.time()

            if state.force_reset:
                print(f"[键盘指令] 重置环境: steps={steps}, reward={episode_reward:.2f}")
                obs, _ = env.reset()
                if hasattr(policy, "reset"):
                    policy.reset()
                episode_reward = 0.0
                steps = 0
                state.force_reset = False
                state.run_policy = False

            if state.run_policy:
                try:
                    obs, reward, terminated, truncated, info = run_policy_step(env, policy, obs, device)
                    episode_reward += float(reward)
                    steps += 1
                except Exception as exc:
                    print(f"物理引擎或策略推理异常: {exc}")
                    terminated, truncated, info = True, False, {}

                if terminated or truncated:
                    reason = "成功" if bool(info.get("is_success", False)) else "结束"
                    print(f"回合{reason}: steps={steps}, reward={episode_reward:.2f}")
                    obs, _ = env.reset()
                    if hasattr(policy, "reset"):
                        policy.reset()
                    episode_reward = 0.0
                    steps = 0
                    state.run_policy = False

            try:
                monitor_frame = render_monitor_frame(
                    sim_env,
                    cfg,
                    state,
                    steps,
                    episode_reward,
                    stereo_matcher,
                    zed_focal_px,
                    zed_baseline_m,
                )
                cv2.imshow(cfg.window_name, monitor_frame)
                handle_cv_key(cv2.waitKey(1) & 0xFF, state)
                render_error_printed = False
            except Exception as exc:
                if not render_error_printed:
                    print(f"相机渲染或 ZED 深度计算失败: {exc}")
                    render_error_printed = True

            if viewer is not None:
                viewer.sync()

            sleep_time = float(cfg.control_dt) - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 主要改这里：权重路径、显示相机、Mujoco 视角、ZED 深度显示范围。
    CONFIG = RunPolicySimConfig(
        ckpt_path="outputs/3_finetune/train/2026-07-09/13-18-11_InsertCylinder-3Arms-v0_ft_zed_diffusion/checkpoints/000004_sr=0.00_reward=327.67_Ploss=-0.0029_Vloss=0.0053",
        display_cameras=(
            "zed_cam_left",
            "zed_cam_right",
            "wrist_cam_left",
            "wrist_cam_right",
            "overhead_cam",
            "worms_eye_cam",
        ),
        show_mujoco_viewer=True,    # 是否显示 Mujoco 自带的 3D 视角，设为 True 可在其中查看物理仿真状态并通过空格键重置环境。
        show_zed_depth=True,        # 是否计算并显示 ZED 立体深度图，设为 True 需要正确配置 zed_left_camera 和 zed_right_camera。
        depth_min_m=0.001,           # ZED 深度图的最小显示距离（单位：米），小于此距离的部分会被裁剪掉以减少噪声影响。
        depth_max_m=0.60,           # ZED 深度图的最大显示距离（单位：米），大于此距离的部分会被裁剪掉以突出近距离物体。
        stereo_num_disparities=256, # 立体匹配器的视差数量，必须是 16 的倍数，增加可以提升远距离深度分辨率但会增加计算量。
        stereo_block_size=3,        # 立体匹配器的块大小，必须是奇数，较小的值可以保留更多细节但可能更噪声，较大的值会更平滑但可能丢失边缘信息。
    )
    run(CONFIG)