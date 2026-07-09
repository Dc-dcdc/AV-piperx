import os
# 训练中评估只设置离屏渲染后端；不要在 import 时强行限线程或禁 TF32。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
import torch
import logging
import numpy as np
import imageio
import json
import shutil
from pathlib import Path
from contextlib import nullcontext
from functools import partial
from tqdm import tqdm
import random
import time
import gymnasium as gym
from lerobot.common.envs.utils import preprocess_observation

# ==========================================
# 🌟 [新增] 自定义 Top-K 快照管理器(包含视频同步清理)
# ==========================================
class TopKCheckpointManager:
    """
    核心逻辑：
    1. 维护一个大小为 max_keep 的列表，按 loss 从小到大排序。
    2. 永远保留最新的 checkpoint（防止训练中断后无法续训最近的进度）。
    3. 自动扫描并删除既不在 top_k 列表，也不是 latest 的多余权重文件夹。
    """
    def __init__(self, out_dir: str, max_keep: int = 5, metric: str = "loss", records_resume: bool = True):
        self.out_dir = Path(out_dir) if out_dir else Path("outputs")
        self.checkpoints_dir = self.out_dir / "checkpoints"  # 模型快照存放的总目录 
        self.eval_dir = self.out_dir / "eval"  # 评估视频存放的总目录
        self.max_keep = max_keep
        self.metric = metric  # 记录筛选指标：'loss' 或 'reward'
        self.top_k = []       # 数据结构: [{"step": int, "loss": float, "path": Path}]
        self.latest_path = None
        self.records_file = self.checkpoints_dir / "top_k_records.json"
        self.records_resume = records_resume
        # 支持断点续训：每次实例化时从本地读取记录，保证跨 step 调用时不丢失历史信息
        if self.records_file.exists() and self.records_resume:
            try:
                with open(self.records_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.latest_path = Path(data.get("latest")) if data.get("latest") else None
                    self.top_k = [
                        {"step": item["step"], "loss": item["loss"], "reward": item["reward"], "path": Path(item["path"])} 
                        for item in data.get("top_k", [])
                    ]
            except Exception as e:
                logging.warning(f"⚠️ 无法读取 Top-K 记录，将重新开始统计: {e}")

    def update(self, step: int, loss: float, ckpt_path: Path,reward: float = -float('inf')):
        self.latest_path = ckpt_path
        # 防重入：如果当前路径已经在记录里了，先剔除旧的
        self.top_k = [item for item in self.top_k if item["path"].name != ckpt_path.name]
        # 将新的 checkpoint 加入 top-k 候选并排序
        self.top_k.append({"step": step, "loss": loss, "reward": reward, "path": ckpt_path})
        
        # 🌟 4. 根据配置进行排序
        if self.metric == "reward":
            # Reward 越大越好，使用 reverse=True
            self.top_k.sort(key=lambda x: x["reward"], reverse=True)
        else:
            # Loss 越小越好
            self.top_k.sort(key=lambda x: x["loss"])
        
        # 如果超出了保留数量，把表现最差的剔除（仅仅是从内存列表中剔除）
        if len(self.top_k) > self.max_keep:
            self.top_k.pop(-1) 
            logging.info(f"候选列表完成 ({len(self.top_k)}/{self.max_keep})，已根据 {self.metric} 剔除表现最差的模型。")
        else:
            logging.info(f"候选列表还在收集中 ({len(self.top_k)}/{self.max_keep})，暂不执行硬盘清理。")
        if self.checkpoints_dir.exists():
            # 提取出合法的【文件夹名称】集合
            valid_names = {item["path"].name for item in self.top_k}
            if self.latest_path:
                valid_names.add(self.latest_path.name) # 最新模型必须保存

            for d in self.checkpoints_dir.iterdir():
                if d.is_dir() and d.name.split('_')[0].isdigit():
                    if d.name not in valid_names:
                        shutil.rmtree(d, ignore_errors=True)
                        logging.info(f"已清理未进入 Top-{self.max_keep} 的模型快照: {d.name}")
            
        
        # 2. 同步清理物理硬盘上的无用评估视频文件夹
        if self.eval_dir.exists():
            # 基于 valid_names 生成合法的视频文件夹名称
            valid_video_folder_names = {f"videos_{name}" for name in valid_names}
            
            for v_dir in self.eval_dir.iterdir():
                # 只清理以 "videos_" 开头，且后缀是数字（或带 loss 的数字）的文件夹
                if v_dir.is_dir() and v_dir.name.startswith("videos_"):
                    # 提取 videos_ 后面的部分判断是不是我们的目标文件夹
                    suffix = v_dir.name.replace("videos_", "")
                    if suffix.split('_')[0].isdigit():
                        if v_dir.name not in valid_video_folder_names:
                            shutil.rmtree(v_dir, ignore_errors=True)
                            logging.info(f"已同步清理失效模型的评估视频: {v_dir.name}")
                            
        self._save_records()

    def _save_records(self):
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        with open(self.records_file, "w", encoding="utf-8") as f:
            json.dump({
                "latest": str(self.latest_path) if self.latest_path else None,
                "top_k": [{"step": i["step"], "loss": i["loss"], "reward": i["reward"], "path": str(i["path"])} for i in self.top_k]
            }, f, indent=4, ensure_ascii=False)

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


def is_vector_env(env) -> bool:
    return int(getattr(env, "num_envs", 1)) > 1


def make_single_eval_env(env_id: str, cameras: list[str], episode_length: int):
    import env as _registered_env  # noqa: F401 - AsyncVectorEnv 子进程里触发 Gym 注册

    env_obj = gym.make(
        id=env_id,
        disable_env_checker=True,
        cameras=cameras,
        episode_length=episode_length,
    )
    return env_obj.unwrapped


def make_eval_env(env_id: str, cameras: list[str], cfg_eval):
    batch_size = int(getattr(cfg_eval, "batch_size", getattr(cfg_eval, "num_envs", 1)))
    n_episodes = int(getattr(cfg_eval, "n_episodes", 1))
    batch_size = max(1, min(batch_size, max(1, n_episodes)))
    episode_length = int(getattr(cfg_eval, "max_steps", 300))

    if batch_size <= 1:
        logging.info("训练评估使用单环境模式。")
        return make_single_eval_env(env_id, cameras, episode_length)

    env_fns = [partial(make_single_eval_env, env_id, cameras, episode_length) for _ in range(batch_size)]
    use_async_envs = bool(getattr(cfg_eval, "use_async_envs", True))
    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    vector_kwargs = {"autoreset_mode": "SameStep"}
    if use_async_envs:
        vector_kwargs.update(shared_memory=True, context="spawn")

    try:
        eval_env = vector_cls(env_fns, **vector_kwargs)
    except TypeError:
        vector_kwargs.pop("autoreset_mode", None)
        try:
            eval_env = vector_cls(env_fns, **vector_kwargs)
        except TypeError:
            eval_env = vector_cls(env_fns)

    mode_name = "AsyncVectorEnv" if use_async_envs else "SyncVectorEnv"
    logging.info(f"训练评估使用多环境模式: {mode_name}, num_envs={batch_size}")
    return eval_env


def prepare_policy_observation(raw_obs: dict, expected_keys: set[str], device) -> dict[str, torch.Tensor]:
    """只转换 policy 需要的键；兼容单环境和 gym.vector batch 观测。"""
    batch = {}

    if "observation.state" in expected_keys:
        state = np.asarray(raw_obs["agent_pos"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if not state.flags.c_contiguous:
            state = np.ascontiguousarray(state)
        batch["observation.state"] = torch.from_numpy(state).to(device, non_blocking=True)

    if "observation.environment_state" in expected_keys and "environment_state" in raw_obs:
        env_state = np.asarray(raw_obs["environment_state"], dtype=np.float32)
        if env_state.ndim == 1:
            env_state = env_state[None, :]
        if not env_state.flags.c_contiguous:
            env_state = np.ascontiguousarray(env_state)
        batch["observation.environment_state"] = torch.from_numpy(env_state).to(device, non_blocking=True)

    pixels = raw_obs.get("pixels", {})
    for key in expected_keys:
        if not key.startswith("observation.images."):
            continue
        camera = key.removeprefix("observation.images.")
        if camera not in pixels:
            raise KeyError(f"环境观测中缺少策略需要的相机: {camera}")

        image = np.asarray(pixels[camera])
        if image.ndim == 3:
            image = image[None, ...]
        if image.ndim != 4:
            raise ValueError(f"相机 {camera} 的观测维度异常: {image.shape}")
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image).to(device, non_blocking=True)
        image_tensor = image_tensor.permute(0, 3, 1, 2).contiguous()
        batch[key] = image_tensor.to(dtype=torch.float32).div_(255.0)

    return batch


def _as_bool_array(value, n_envs: int, default: bool = False) -> np.ndarray:
    if value is None:
        return np.full(n_envs, default, dtype=bool)
    arr = np.asarray(value)
    if arr.shape == ():
        return np.full(n_envs, bool(arr), dtype=bool)
    arr = arr.astype(bool).reshape(-1)
    if arr.shape[0] < n_envs:
        padded = np.full(n_envs, default, dtype=bool)
        padded[: arr.shape[0]] = arr
        return padded
    return arr[:n_envs]


def _as_float_array(value, n_envs: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(n_envs, default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == ():
        return np.full(n_envs, float(arr), dtype=np.float32)
    arr = arr.reshape(-1)
    if arr.shape[0] < n_envs:
        padded = np.full(n_envs, default, dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded
    return arr[:n_envs]


def _extract_info_bool(info, key: str, n_envs: int, default: bool = False) -> np.ndarray:
    values = np.full(n_envs, default, dtype=bool)
    if not isinstance(info, dict):
        return values

    def merge(raw_value, raw_mask):
        parsed = _as_bool_array(raw_value, n_envs, default=default)
        mask = _as_bool_array(raw_mask, n_envs, default=True)
        values[mask] = parsed[mask]

    if key in info:
        merge(info[key], info.get(f"_{key}", np.ones(n_envs, dtype=bool)))

    final_infos = info.get("final_info", None)
    if final_infos is not None:
        final_mask = _as_bool_array(
            info.get("_final_info", np.ones(n_envs, dtype=bool)),
            n_envs,
            default=True,
        )

        # Gymnasium 新版 VectorEnv 将 final_info 合并为 dict-of-arrays。
        if isinstance(final_infos, dict):
            if key in final_infos:
                key_mask = _as_bool_array(
                    final_infos.get(f"_{key}", np.ones(n_envs, dtype=bool)),
                    n_envs,
                    default=True,
                )
                merge(final_infos[key], final_mask & key_mask)
        else:
            # 兼容旧版 object array/list-of-dicts。
            for idx, final_info in enumerate(list(final_infos)[:n_envs]):
                if final_mask[idx] and isinstance(final_info, dict) and key in final_info:
                    values[idx] = bool(final_info[key])

    return values


def _render_vector_frames(env, camera: str, n_envs: int):
    if hasattr(env, "call"):
        frames = env.call("render", [camera])
        return list(frames)
    return [env.envs[idx].unwrapped.render([camera]) for idx in range(n_envs)]


def custom_eval_policy_vectorized(env, policy, cfg_eval, videos_dir, device):
    """训练中用 VectorEnv 并行跑评估 episode，保持 policy action chunk 队列按 batch 同步推进。"""
    policy.eval()
    n_envs = int(env.num_envs)
    successes = []
    rewards = []
    episode_records = []
    saved_video_paths = []

    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = int(getattr(cfg_eval, "n_episodes", 10))
    max_rendered = int(getattr(cfg_eval, "max_episodes_rendered", 4))
    fps = int(getattr(cfg_eval, "fps", 25))
    max_steps = int(getattr(cfg_eval, "max_steps", 300))
    raw_camera = getattr(cfg_eval, "render_camera", "overhead_cam")
    render_cameras = [raw_camera] if isinstance(raw_camera, str) else list(raw_camera)
    expected_keys = set(policy.config.input_shapes)
    base_seed = int(getattr(cfg_eval, "seed", 1000))
    global_real_inference_times = []

    episode_cursor = 0
    progress = tqdm(total=n_episodes, leave=False)
    while episode_cursor < n_episodes:
        active_count = min(n_envs, n_episodes - episode_cursor)
        episode_ids = np.arange(episode_cursor, episode_cursor + active_count, dtype=np.int64)
        active = np.zeros(n_envs, dtype=bool)
        active[:active_count] = True
        seeds = [
            base_seed + int(episode_ids[idx]) if idx < active_count else base_seed + n_episodes + idx
            for idx in range(n_envs)
        ]

        seed_runtime(int(seeds[0]))
        try:
            obs, _ = env.reset(seed=seeds)
        except TypeError:
            logging.warning("当前 VectorEnv 不支持列表 seed，回退为无 seed reset。")
            obs, _ = env.reset()

        policy.reset()
        seed_runtime(int(seeds[0]))

        done = np.zeros(n_envs, dtype=bool)
        episode_success = np.zeros(n_envs, dtype=bool)
        episode_rewards = np.zeros(n_envs, dtype=np.float32)
        steps_taken = np.zeros(n_envs, dtype=np.int32)
        frames_by_env = {
            idx: {camera: [] for camera in render_cameras}
            for idx in range(active_count)
            if int(episode_ids[idx]) < max_rendered
        }
        render_failed_cameras = set()
        batch_inference_times = []

        for step in range(max_steps):
            step_active = active & ~done
            if not step_active.any():
                break

            if frames_by_env:
                render_indices = [idx for idx in frames_by_env if step_active[idx]]
                if render_indices:
                    for camera in render_cameras:
                        if camera in render_failed_cameras:
                            continue
                        try:
                            frames = _render_vector_frames(env, camera, n_envs)
                        except Exception as exc:
                            render_failed_cameras.add(camera)
                            logging.warning(f"VectorEnv 渲染相机 {camera} 失败，本批次跳过该视角录像: {exc}")
                            continue
                        for env_idx in render_indices:
                            frames_by_env[env_idx][camera].append(frames[env_idx])

            batch_obs = prepare_policy_observation(obs, expected_keys, device)
            start_time = time.perf_counter()
            with torch.no_grad():
                action = policy.select_action(batch_obs)
            action_np = action.detach().cpu().numpy().copy()
            if action_np.ndim == 1:
                action_np = action_np[None, :]
            if action_np.shape[0] != n_envs:
                raise ValueError(f"策略输出 batch={action_np.shape[0]}，但评估环境 num_envs={n_envs}")
            action_np[~step_active] = 0.0

            inference_time_ms = (time.perf_counter() - start_time) * 1000
            batch_inference_times.append(inference_time_ms)
            steps_taken[step_active] = step + 1

            try:
                obs, reward, terminated, truncated, info = env.step(action_np)
            except Exception as exc:
                logging.exception(f"VectorEnv 物理引擎崩溃 (Step {step})，本批未完成 episode 记为失败: {exc}")
                unfinished = active & ~done
                episode_rewards[unfinished] = -1000.0
                episode_success[unfinished] = False
                done[unfinished] = True
                break

            reward_arr = _as_float_array(reward, n_envs)
            terminated_arr = _as_bool_array(terminated, n_envs)
            truncated_arr = _as_bool_array(truncated, n_envs)
            step_done = terminated_arr | truncated_arr | (step + 1 >= max_steps)
            success_arr = _extract_info_bool(info, "is_success", n_envs, default=False)

            episode_rewards[step_active] += reward_arr[step_active]
            newly_done = step_active & step_done
            episode_success[newly_done] = success_arr[newly_done]
            done[newly_done] = True

        unfinished = active & ~done
        if unfinished.any():
            steps_taken[unfinished] = np.maximum(steps_taken[unfinished], max_steps)
            episode_success[unfinished] = False
            done[unfinished] = True

        for local_idx in range(active_count):
            ep = int(episode_ids[local_idx])
            success = bool(episode_success[local_idx])
            ep_reward = float(episode_rewards[local_idx])
            successes.append(success)
            rewards.append(ep_reward)
            episode_records.append(
                {
                    "episode": ep,
                    "seed": base_seed + ep,
                    "success": success,
                    "reward": ep_reward,
                    "steps": int(steps_taken[local_idx]),
                }
            )

            if local_idx in frames_by_env:
                status = "Success" if success else "Fail"
                for camera, frames in frames_by_env[local_idx].items():
                    if len(frames) == 0:
                        continue
                    video_name = f"{camera}_ep_{ep}_reward={ep_reward:.1f}_{status}.mp4"
                    video_path = videos_dir / video_name
                    imageio.mimsave(str(video_path), frames, fps=fps)
                    saved_video_paths.append(str(video_path))

        real_inferences = [t for t in batch_inference_times if t > 5.0]
        if real_inferences:
            global_real_inference_times.extend(real_inferences)

        episode_cursor += active_count
        progress.update(active_count)

    progress.close()

    if len(global_real_inference_times) > 0:
        warmup_steps = 3
        stable_times = (
            global_real_inference_times[warmup_steps:]
            if len(global_real_inference_times) > warmup_steps
            else global_real_inference_times
        )
        max_time = max(stable_times)
        avg_real_time = sum(stable_times) / len(stable_times)
    else:
        avg_real_time = 0.0
        max_time = 0.0

    policy.train()
    return {
        "aggregated": {
            "success_rate": float(np.mean(successes)),
            "average_reward": float(np.mean(rewards)),
            "avg_inference_ms": float(avg_real_time),
            "max_inference_ms": float(max_time),
        },
        "video_paths": saved_video_paths,
        "episodes": episode_records,
    }

def custom_eval_policy(env, policy, cfg_eval, videos_dir, device):
    """
    完全自主实现的评估代码。没有任何黑盒。
    接收标准 Gym 环境，处理图像归一化，跑策略推理，保存视频。
    """
    if is_vector_env(env):
        return custom_eval_policy_vectorized(env, policy, cfg_eval, videos_dir, device)

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

            def prepare_obs(obj):
                """递归字典，拷贝连续内存，并强行在最前面增加一个 Batch 维度"""
                if isinstance(obj, dict):
                    return {k: prepare_obs(v) for k, v in obj.items()}
                elif hasattr(obj, "copy"):  # 如果是 numpy 数组
                    return np.expand_dims(obj.copy(), axis=0).copy() # [H, W, C] -> [1, H, W, C]
                return obj
            
            # 在送入官方预处理之前，强制清洗内存并扩维
            obs = prepare_obs(obs)

            # [b, H, W, C] -> [b, C, H, W] ,并/ 255.0
            obs = preprocess_observation(obs)

            # 2. 键值过滤与设备转移：只保留模型配置中真正需要的输入特征，推入 GPU
            obs = {
                k: v.to(device)
                for k, v in obs.items()
                if k in policy.config.input_shapes  # 🌟 保留这层保护，防止多余状态引发报错
            }
            # ==========================================
            # ⏱️ 开始计时：使用高精度的 perf_counter
            # ==========================================
            start_time = time.perf_counter()

            # 3. 推理获取动作
            with torch.no_grad():
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
    policy.train() # 恢复训练模式
    
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


def evaluate_and_checkpoint_if_needed(
    step, policy, optimizer, lr_scheduler, logger, cfg, device, out_dir, eval_env=None, train_loss=None, manager=None
):
    """
    主评估与保存入口
    """
    _num_digits = max(6, len(str(cfg.training.offline_steps))) 
    step_identifier = f"{step:0{_num_digits}d}" 

    if train_loss is not None:
        base_identifier = f"{step_identifier}_loss={train_loss:.4f}"
    else:
        base_identifier = step_identifier
    final_identifier = base_identifier
    temp_video_dir = None
    ar = -float("inf")
    
    # 1. 评估逻辑 (优先读取 cfg.eval.eval_freq)
    eval_freq = getattr(cfg.training, "eval_freq", 0)
    is_last_step = (step == cfg.training.offline_steps - 1)
    # 初始化本步的 reward 为极小值
    sr = -float('inf')
    if (eval_freq > 0 and step > 0 and step % eval_freq == 0) or is_last_step:
        logging.info(f"开始自主评估流程, 当前 Step: {step}")
        if eval_env is not None:
            temp_video_dir = Path(out_dir) / "eval" / f"videos_{base_identifier}"
            
            with torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                # 传入完整的 cfg.eval 节点
                eval_info = custom_eval_policy(
                    env=eval_env,
                    policy=policy,
                    cfg_eval=cfg.eval, 
                    videos_dir=temp_video_dir,
                    device=device
                )
            
            sr = eval_info["aggregated"]["success_rate"]
            ar = eval_info["aggregated"]["average_reward"]
            avg_infer = eval_info["aggregated"]["avg_inference_ms"]
            max_infer = eval_info["aggregated"]["max_inference_ms"]
            logging.info(f"评估完毕! 成功率: {sr*100:.1f}%, 平均奖励: {ar:.2f}, 推理平均耗时: {avg_infer:.2f} ms， 推理最大耗时: {max_infer:.2f} ms")

            if getattr(cfg, "wandb", {}).get("enable", False) and len(eval_info["video_paths"]) > 0:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval") # 只上传第一个视频到 wandb

            final_identifier = f"{base_identifier}_sr={sr*100:.1f}_ar={ar:.2f}"

    save_freq = getattr(cfg.training, "save_freq", 10000)
    if getattr(cfg.training, "save_checkpoint", False) and (step > 0 and step % save_freq == 0 or is_last_step):
        # 保存模型权重
        logging.info(f"保存模型快照... Step: {step}")
        logger.save_checkpoint(step, policy, optimizer, lr_scheduler, identifier=final_identifier)

        # 归档评估视频
        ckpt_path = Path(out_dir) / "checkpoints" / final_identifier
        if ckpt_path.exists() and temp_video_dir is not None and temp_video_dir.exists():
            target_video_path = ckpt_path / "eval_videos"
            # 使用 shutil.move 将整个临时文件夹移动并重命名为 eval_videos
            shutil.move(str(temp_video_dir), str(target_video_path))
            logging.info(f"视频已归档至: {final_identifier}/eval_videos/")

        # 触发 Top-K 筛选与清理
        if train_loss is not None:
            if ckpt_path.exists() and manager is not None:
                manager.update(step, train_loss, ckpt_path, reward=ar)
            elif manager is None:
                logging.warning("⚠️ 警告: 未传入 TopKCheckpointManager，跳过 Top-K 模型清理逻辑。")
        else:
            logging.warning("⚠️ 警告: 未传入 train_loss，跳过 Top-K 模型清理逻辑，将保留所有权重。")
