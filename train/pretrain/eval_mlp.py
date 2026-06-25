import os
# 告诉底层的 CuBLAS 使用固定的工作区配置，以保证计算的绝对确定性
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 🌟 终极防线：强行限制底层所有的数学库和物理引擎仅使用单线程！
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# 独立评估优先使用 EGL 离屏渲染，避免 GLFW/X11 窗口后端带来的额外波动。
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import logging
import numpy as np
import imageio
import json
import shutil
from pathlib import Path
from contextlib import nullcontext, redirect_stdout
import gymnasium as gym
import yaml
from tqdm import tqdm
import random
import time
import sys
import contextlib
from types import SimpleNamespace
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import get_safe_torch_device, init_logging
from lerobot.common.envs.utils import preprocess_observation

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import env.task.sim_envs
from train.finetune.finetune_mlp import FrozenDiffusionMLPPolicy


@contextlib.contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull):
            yield


def load_frozen_diffusion_mlp_policy(base_policy, load_dir: str | Path, device):
    """加载 residual MLP 微调权重。"""
    load_dir = Path(load_dir)
    mlp_ckpt_path = load_dir / "residual_mlp.pt"
    mlp_config_path = load_dir / "residual_mlp_config.json"
    if not mlp_ckpt_path.exists():
        mlp_ckpt_path = load_dir / "action_mlp.pt"
        mlp_config_path = load_dir / "action_mlp_config.json"

    if not mlp_ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到 residual_mlp.pt 或 action_mlp.pt: {load_dir}\n"
            "请确认 ckpt_path 指向 residual MLP 微调保存的 checkpoint 或其 pretrained_model 子目录。"
        )

    try:
        mlp_ckpt = torch.load(mlp_ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        mlp_ckpt = torch.load(mlp_ckpt_path, map_location=device)
    mlp_json = {}
    if mlp_config_path.exists():
        with open(mlp_config_path, "r", encoding="utf-8") as f:
            mlp_json = json.load(f)

    def pick(key, default):
        if key in mlp_ckpt:
            return mlp_ckpt[key]
        return mlp_json.get(key, default)

    def pick_list(key, default):
        value = pick(key, default)
        if value is None:
            return []
        return [int(item) for item in list(value)]

    action_dim = int(pick("action_dim", base_policy.config.output_shapes["action"][0]))
    action_start = int(pick("action_start", getattr(base_policy.config, "n_obs_steps", 2) - 1))
    action_end = int(
        pick(
            "action_end",
            action_start + getattr(base_policy.config, "n_action_steps", base_policy.config.n_action_steps),
        )
    )
    hidden_dim = int(pick("hidden_dim", pick("residual_mlp_hidden_dim", 256)))
    depth = int(pick("depth", pick("residual_mlp_depth", 2)))
    learn_std = bool(pick("learn_std", pick("residual_mlp_learn_std", True)))
    logprob_reduction = str(pick("logprob_reduction", "mean"))
    default_arm_indices = list(range(min(14, action_dim)))
    default_cam_indices = list(range(14, action_dim)) if action_dim > 14 else []
    arm_action_indices = pick_list("arm_action_indices", default_arm_indices)
    cam_action_indices = pick_list("cam_action_indices", default_cam_indices)
    lambda_arm = float(pick("lambda_arm", 1.0))
    lambda_cam = float(pick("lambda_cam", 1.0))
    arm_hidden_dim = int(pick("arm_hidden_dim", hidden_dim))
    cam_hidden_dim = int(pick("cam_hidden_dim", hidden_dim))
    arm_depth = int(pick("arm_depth", depth))
    cam_depth = int(pick("cam_depth", depth))

    log_std = mlp_ckpt.get("log_std", torch.full((action_dim,), np.log(0.02), dtype=torch.float32))
    init_std = float(log_std.float().exp().mean().item())
    if "mlp_arm" not in mlp_ckpt or "mlp_cam" not in mlp_ckpt:
        raise ValueError(
            f"{mlp_ckpt_path} 不是当前 residual MLP 权重格式，缺少 mlp_arm/mlp_cam。"
        )

    policy = FrozenDiffusionMLPPolicy(
        base_policy=base_policy,
        action_dim=action_dim,
        action_start=action_start,
        action_end=action_end,
        hidden_dim=hidden_dim,
        depth=depth,
        init_std=init_std,
        learn_std=learn_std,
        logprob_reduction=logprob_reduction,
        arm_indices=arm_action_indices,
        cam_indices=cam_action_indices,
        lambda_arm=lambda_arm,
        lambda_cam=lambda_cam,
        arm_hidden_dim=arm_hidden_dim,
        cam_hidden_dim=cam_hidden_dim,
        arm_depth=arm_depth,
        cam_depth=cam_depth,
    ).to(device)
    if policy.arm_action_indices:
        policy.mlp_arm.load_state_dict(mlp_ckpt["mlp_arm"], strict=True)
    if policy.cam_action_indices:
        policy.mlp_cam.load_state_dict(mlp_ckpt["mlp_cam"], strict=True)
    with torch.no_grad():
        policy.log_std.copy_(log_std.to(device))
    policy.eval()

    logging.info(
        f"已加载 DP+ResidualMLP 微调模型: action_dim={action_dim}, action_slice=[{action_start}:{action_end}], "
        f"arm_indices={policy.arm_action_indices}, cam_indices={policy.cam_action_indices}, "
        f"lambda_arm={policy.lambda_arm:.3f}, lambda_cam={policy.lambda_cam:.3f}, "
        f"mean_action_std={float(policy.action_std.mean().detach().cpu().item()):.5f}"
    )
    return policy

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
    render_camera = raw_camera
    # 从配置中提取基础种子，默认为 1000
    base_seed = getattr(cfg_eval, "seed", 1000)
    quiet = bool(getattr(cfg_eval, "quiet", True))
    show_progress = bool(getattr(cfg_eval, "show_progress", not quiet))
    suppress_model_stdout = bool(getattr(cfg_eval, "suppress_model_stdout", quiet))
    # 计算所有评估回合推理的总耗时
    global_real_inference_times = []
    # 动作执行循环
    episode_iter = tqdm(
        range(n_episodes),
        desc="Eval episodes",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for ep in episode_iter:
        # 计算当前回合的专属种子，并传给环境
        ep_seed = base_seed + ep
        # 先固定全局 RNG，再 reset 环境；这样 reset 内部若用 random/np/torch 也可复现。
        seed_runtime(ep_seed)
        seed_env_spaces(env, ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        done = False
        frames = []
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
                frames.append(env.unwrapped.render(render_camera)) # gym创建需要加上 .unwrapped
                # frames.append(env.render(render_camera))  #直接创建环境不需要

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
                with maybe_suppress_stdout(suppress_model_stdout):
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
        if ep < max_rendered and len(frames) > 0:
            status = "Success" if successes[ep] else "Fail"
            # 格式：012500_reward=150.5_cam_overhead_ep_0.mp4
            video_name = f"{render_camera[0]}_ep_{ep}_reward={ep_reward:.1f}_{status}.mp4"
            video_path = videos_dir / video_name
            imageio.mimsave(str(video_path), frames, fps=fps)
            # logging.info(f"  保存视频: {video_path.name}")
            
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
    
    # 计算实际应该返回的视频列表长度
    actual_rendered = min(max_rendered, n_episodes)
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
            if ckpt_path.exists():
                manager.update(step, train_loss, ckpt_path, reward=ar)
        else:
            logging.warning("⚠️ 警告: 未传入 train_loss，跳过 Top-K 模型清理逻辑，将保留所有权重。")


def seed_everything(seed: int):
    """Set seed for absolute reproducibility."""
    
    # 1. 锁死 Python 字典和集合的哈希种子 
    # (防止字典遍历顺序在不同运行中发生变化，导致 Batch 数据错位)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. 锁死 Python / Numpy / PyTorch (CPU & 所有 GPU)
    seed_runtime(seed)

    # 3. 锁死线程数，减少 CPU 规约/物理仿真侧的调度差异
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads 只能在并行工作开始前调用；已开始时跳过即可。
        pass

    # 4. 锁死 CuDNN 底层算法 (极其关键)
    # 关闭自动调优，强行使用确定性卷积算法
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

    # 5. 强制 PyTorch 使用确定性算子
    # 如果底层调用了非确定性算子，会强制回退到确定性实现，或者抛出警告
    torch.use_deterministic_algorithms(True, warn_only=True)

def ensure_python_hash_seed(seed: int):
    """PYTHONHASHSEED 必须在解释器启动前生效；独立运行时自动重启一次补齐。"""
    desired = str(seed)
    if os.environ.get("PYTHONHASHSEED") == desired:
        return

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = desired
    logging.warning(f"PYTHONHASHSEED 需要在 Python 启动前设置，正在用 PYTHONHASHSEED={desired} 自动重启 eval_mlp.py。")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

def main(eval_cfg):
    ckpt_path = eval_cfg.ckpt_path

    seed_everything(eval_cfg.seed)
    patch_act_position_embedding_for_determinism()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ 找不到权重路径: {ckpt_path}\n请检查路径是否正确。")
    # ==========================================
    # 🌟 1.自动探测 LeRobot 的 pretrained_model 子文件夹
    # ==========================================
    hf_model_dir = os.path.join(ckpt_path, "pretrained_model")
    if os.path.exists(hf_model_dir):
        logging.info(f"检测到 LeRobot 标准快照结构，将自动读取子目录: pretrained_model")
        load_dir = hf_model_dir
    else:
        load_dir = ckpt_path # 兼容其他保存格式

    # 准备设备；如需最强确定性，可在 eval_cfg 中设为 "cpu"。
    device = get_safe_torch_device(getattr(eval_cfg, "device", "cuda"))
    logging.info(f"初始化评估程序... 使用设备: {device}")

    
    # ==========================================
    # 🌟 2.实例化 Policy 并加载权重
    # ==========================================
    logging.info(f"正在从目录重建网络并加载权重: {load_dir}")
    try:
        
        from pathlib import Path
        from lerobot.common.utils.utils import init_hydra_config

        # 1. 寻找 config.yaml 路径
        config_yaml_path = Path(load_dir) / "config.yaml"
        if not config_yaml_path.exists():
            config_yaml_path = Path(load_dir).parent / "config.yaml"
            
        if not config_yaml_path.exists():
            raise FileNotFoundError(f"找不到 config.yaml，无法初始化 hydra_cfg！")

        # 2. 根据 yaml 文件初始化 hydra_cfg 对象
        hydra_cfg = init_hydra_config(str(config_yaml_path))
        hydra_cfg.device = str(device)
        # if hydra_cfg.policy.name == "diffusion" and "do_mask_loss_for_padding" not in hydra_cfg.policy:
        #     hydra_cfg.policy.do_mask_loss_for_padding = False

        # 3. 🌟 核心：直接使用 make_policy，让框架接管底层张量与 EMA 加载
        policy = make_policy(
            hydra_cfg=hydra_cfg, 
            pretrained_policy_name_or_path=str(load_dir)
        )
        
        logging.info("  成功使用 make_policy 加载冻结 Diffusion 底座！底层 Normalizer 与平滑权重已自动生效。")
        policy.to(device)
        policy = load_frozen_diffusion_mlp_policy(
            base_policy=policy,
            load_dir=load_dir,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f"❌ 权重加载失败！详细报错: {e}")
    
    # ==========================================
    # 🌟 3.读取快照中的配置，使环境和训练时的对齐
    # ==========================================
    all_obs_keys = policy.config.input_shapes.keys()
    ref_cams = [k.replace("observation.images.", "") for k in all_obs_keys if "observation.images." in k]
    if not ref_cams:
        raise ValueError(f"❌ 严重冲突：模型中未找到相机相关参数。请检查模型输入是否正确。")
    
    obs_cameras = list(dict.fromkeys(ref_cams + eval_cfg.render_camera))
    # 动态读取环境元数据 (直接从 load_dir 读取)
    
    config_yaml_path = Path(load_dir) / "config.yaml"

    if config_yaml_path.exists():
        with open(config_yaml_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            
            # 安全地从 YAML 的字典树中提取 env.name 和 env.task
            env_cfg = full_cfg.get("env", {})
            env_name = env_cfg.get("name", getattr(env_cfg, "name", "guided_vision"))
            env_task = env_cfg.get("task", getattr(env_cfg, "task", "SewNeedle-3Arms-v0"))
            logging.info(f"成功从预训练文件夹读取完整环境配置: {env_name}/{env_task}")
    else:
        # 极限防呆后备
        env_name = getattr(eval_cfg, "name", "guided_vision")
        env_task = getattr(eval_cfg, "task", "SewNeedle-3Arms-v0")
        logging.warning(f"  未找到 config.yaml，使用本地设定的后备环境: {env_name}/{env_task}")

    # 拼接 Gym ID
    env_id = f"{env_name}/{env_task}"
    logging.info(f"正在通过 Gym 注册表构建环境: {env_id}")
    # 使用 gym.make 创建环境，并通过 kwargs 强行覆盖你需要的相机
    eval_env = gym.make(
        id=env_id, 
        cameras=obs_cameras  # 👈 这里的传参会直接覆盖 __init__.py 里的默认套餐！
    )
    env_desc = f"{env_id} -> {eval_env.unwrapped.__class__.__module__}.{eval_env.unwrapped.__class__.__name__}"
    logging.info(f"当前评估环境: {env_desc}")
    # print(f"当前评估环境: {env_desc}", flush=True)
    logging.info(f"环境加载成功！最终挂载的相机: {obs_cameras}")

    # ==========================================
    # 🌟 4.设置视频输出目录 (默认在 000000 文件夹同级新建 eval_videos 文件夹)
    # ==========================================
    videos_dir = os.path.join(ckpt_path, "extra_eval_videos")
    logging.info(f"开始测试! 录像将保存在: {videos_dir}")

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
    # 🌟 6.整理指标，重命名文件夹归档视频
    # ==========================================
    import shutil
    sr = eval_info["aggregated"]["success_rate"]
    ar = eval_info["aggregated"]["average_reward"]
    avg_infer = eval_info["aggregated"]["avg_inference_ms"]
    max_infer = eval_info["aggregated"]["max_inference_ms"]
    # 生成直观的文件夹名称 (例如: eval_seed100_sr85.0_ar150.5)
    new_folder_name = f"eval_seed={eval_cfg.seed}_ep={eval_cfg.n_episodes}_sr={sr*100:.1f}_ar={ar:.2f}"
    new_videos_dir = os.path.join(videos_dir, new_folder_name)
    os.makedirs(new_videos_dir, exist_ok=True)
    
    # 将刚刚生成的视频全部移动到新文件夹中
    moved_count = 0
    for video_path in eval_info["video_paths"]:
        if os.path.exists(video_path):
            file_name = os.path.basename(video_path)
            shutil.move(video_path, os.path.join(new_videos_dir, file_name))
            moved_count += 1

    episode_results_path = os.path.join(new_videos_dir, "episode_results.json")
    with open(episode_results_path, "w", encoding="utf-8") as f:
        json.dump(eval_info["episodes"], f, indent=2, ensure_ascii=False)
    logging.info(f"逐 episode 评估明细已保存: {episode_results_path}")

    # ==========================================
    # 🌟 7.打印最终结果
    # ==========================================
    logging.info("="*50)
    logging.info(f"--独立评估完成！")
    logging.info(f"--成功率 (Success Rate): {sr*100:.1f}%")
    logging.info(f"--平均奖励 (Average Reward): {ar:.2f}")
    logging.info(f"--平均推理时间 (Average Inference Time): {avg_infer:.2f} ms")
    logging.info(f"--最大推理时间 (Max Inference Time): {max_infer:.2f} ms")
    logging.info("="*50)

# =========================================================================
# 🌟 独立评估测试入口，用于评估 test_finetune.py 保存的 DP+MLP checkpoint
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
        # 📂 模型路径设置：指向 checkpoint 文件夹即可，代码会自动寻找 residual_mlp.pt。
        ckpt_path="outputs/3_finetune/train/2026-06-02/00-06-22_SewNeedle-2Arms-v0_ft_wrist_diffusion_mlp/checkpoints/000014_sr=0.75_reward=488.13_MLPloss=0.1031_Vloss=1.6439/pretrained_model",
        # ⚙️ 评估参数设置
        n_episodes=100,             # 评估多少个任务                 
        max_episodes_rendered=0,  # 保存多少个视频 
        fps=25,                   # 视频帧率，和环境控制频率对齐
        max_steps=300,            # 每个任务的最大步数
        device="cuda",            # 如需完全规避 CUDA 非确定算子，可临时改成 "cpu"
        
        # 📷 相机设置
        # ['zed_cam_left', 'zed_cam_right', 'overhead_cam', 'worms_eye_cam' , 'wrist_cam_left', 'wrist_cam_right'],
        render_camera=['overhead_cam'],         # 保存video的相机视角    
        # ⚡ 混合精度设置 (推荐在评估时设为 False 以保证确定性)
        use_amp=False,          
        # 🧹 终端显示设置：默认安静评估，只输出关键日志和最终结果。
        quiet=True,
        show_progress=False,
        suppress_model_stdout=True,
    )
    ensure_python_hash_seed(eval_cfg.seed)
    # ==========================================print("cuda available:", torch.cuda.is_available())
    # 启动
    main(eval_cfg=eval_cfg)
                                         
