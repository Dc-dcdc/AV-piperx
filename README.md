# DPPO
先用扩散模型预训练并使用强化学习PPO算法进行微调
## ✨ Pretrain部分 
1. 训练代码位于 `train/pretrain/train_pretrain.py`，训练对应的配置参数位于`configs/pretrain/policy`，训练不同的任务时，需要注意修改`train_pretrain.py`中的`env`(决定场景)和`policy`(决定训练策略：ACT、diffusion)。
2. 模型快照和评估视频储存的默认位置位于`configs/pretrain/pre_default.yaml`中的hydra.run.dir,这是相对于该项目（DPPO）的相对保存位置，注意命令行的运行位置，否则会存到其他地方，wandb的保存文件名为hydra.run.job。
3. 预训练代码设置了断点续训，输入保存的模型快照路径并设置`resume=true`即可，会自动读取训练时使用的policy配置参数。
4. 评估代码位于`train/pretrain/eval.py`，输入模型快照的路径即可，会自动读取训练时使用的policy配置参数，可以在`eval.py`设置`render_camera=['overhead_cam']`来设置录制视频的视角。
5. 值得注意的是，这里用到了av-aloha的lerobot代码，换设备训练需要注意，后期可以注意更新为官网版本的lerobot
## ✨ Finetune部分 
1. 微调代码位于`train/finetune/train_finetune.py`，输入模型快照的路径即可，会自动读取训练时使用的policy配置参数
2. 为了适配评估代码`train/pretrain/eval.py`，保存权重的同时生成了对应的训练参数配置表`config.yaml`和`config.json`


## ✨ sim_env部分 
1. 添加了模型推理部分，可以添加训练好的模型进行在线仿真推理，可以修改display_cameras参数获取要单独渲染的相机视角，一行两个进行排布。此外还可以通过修改代码中SIM_DT为具体值，从而实现慢速的观测效果。


## ✨ data_collect部分 
### 1. 脚本说明
| 文件/目录 | 作用 | 关键参数或输入输出 |
|---|---|---|
| `data_collect/quest_teleop_collect.py` | Quest3 遥操采集主程序，接收 Quest 位姿、映射到机器人动作、执行环境并保存成功轨迹 | 配置文件：`configs/data_collect/quest_teleop_collect.yaml`；`env_id` 选择环境；`record_cameras` 选择保存相机；`save_pose_action` 是否保存末端位姿动作；`save_videos` 是否保存相机视频；`head_control` 是否启用头显控制中间臂；`lock_roll/lock_pitch` 是否锁定中间臂姿态轴 |
| `data_collect/quest_mujoco_test.py` | Quest3 到 MuJoCo 的在线遥操测试，不负责正式保存数据 | 配置文件：`configs/data_collect/quest_mujoco_test.yaml`；`env_id` 选择环境；`display_camera` 选择显示/发送相机；`unity_image_stream` 是否向 Quest 发送画面；`hand_position_scale/head_position_scale` 控制位移映射比例 |
| `data_collect/quest_pose_mapping_test.py` | 只测试 Quest 位姿到 MuJoCo 坐标系的映射关系，适合排查坐标轴和姿态方向 | 输入 Quest UDP 数据；输出可视化位姿映射结果 |
| `data_collect/collect_data_from_model.py` | 使用已经训练好的模型在仿真中自动收集成功轨迹 | `CKPT_PATH` 模型路径；`OUTPUT_DIR` 原始数据输出目录；`MAX_STEPS` 单条轨迹最大步数；`TARGET_SUCCESSES` 目标成功轨迹数；`SAVE_VIDEOS` 是否生成视频 |
| `data_collect/quest_receive.py` | 接收 Quest/Unity 通过 UDP 发送的头显和手柄原始数据 | 输入：UDP JSON；输出：`HeadsetData` |
| `data_collect/quest_control.py` | 将 Quest 头显和手柄位姿映射为三臂末端位姿动作 | 输入：`HeadsetData` 和机器人当前末端位姿；输出：`pose_action` |
| `data_collect/robot_ik_solver.py` | 将三臂末端位姿动作转换为关节动作 | 输入：`pose_action`；输出：`joint_action`，可直接用于 `env.step` |
| `data_collect/quest_send.py` | 将 MuJoCo 渲染画面通过 UDP 分片发送给 Unity/Quest 显示 | `unity_image_host` 接收端 IP；`unity_image_port` 接收端端口；`unity_image_hz` 发送频率；`unity_image_jpeg_quality` JPEG 压缩质量 |
| `data_collect/headset_utils.py` | 定义 Quest 数据结构和按键/追踪状态工具 | 主要结构：`HeadsetData`、`HeadsetFeedback` |
| `data_collect/transform_utils.py` | 存放坐标系转换、位姿矩阵、四元数等工具函数 | 用于 Unity/Quest 坐标系和 MuJoCo 坐标系之间的转换 |
| `data_collect/meta quest3` | Unity Quest3 工程，包含发送 Quest 位姿和接收 MuJoCo 图像的脚本 | 需要在 Unity/Quest 端配置 UDP IP、端口和显示位置 |

### 2. Quest 遥操数据流
```text
Quest/Unity
  -> quest_receive.py
  -> quest_control.py
  -> robot_ik_solver.py
  -> env.step(joint_action)
  -> quest_teleop_collect.py 保存成功轨迹
```
其中：
```text
Quest 原始位姿       -> HeadsetData
HeadsetData          -> pose_action      # 三臂末端位姿动作
pose_action          -> joint_action     # 三臂关节动作
joint_action         -> env.step
```
保存数据时，`joint_action` 始终保存；如果 `save_pose_action=true`，会额外保存 `pose_action`，方便后续训练末端位姿策略。

### 3. 遥操采集输出
默认配置：
```text
configs/data_collect/quest_teleop_collect.yaml
```
默认输出目录：
```text
outputs/4_data_collect/quest_teleop/quest_teleop_SewNeedle-3Arms-v0
```
关键配置：
```text
record_cameras      # 保存哪些相机视频，需和训练配置 input_shapes 对齐
save_videos         # 是否保存每条 episode 的相机 mp4
save_pose_action    # 是否额外保存末端位姿动作
fps                 # 数据时间戳和相机视频帧率
max_steps_per_episode  # 单条 episode 最大采集步数
```
保存格式：
```text
metadata.json
episodes/
  episode_000000/
    info.json
    arrays.npz
      joint_action
      pose_action
      observation_state
    videos/
      <camera>.mp4
```
采集脚本只保存任务成功并经过 A+X 确认的轨迹；转换为 LeRobot/HF 数据集见下一节 `hugging face部分`。

## ✨ hugging face部分 
### 1. 脚本说明
| 文件 | 作用 | 关键参数 |
|---|---|---|
| `hugging_face/convert_data_to_hf.py` | 将本地采集的 raw 数据转换为 LeRobot/HF 数据集格式 | `RAW_DIR` 原始数据目录；`OUTPUT_DIR` 转换后目录；`ACTION_KEY` 选择 `joint_action` 或 `pose_action` 映射为 LeRobot 的 `action` |
| `hugging_face/push_data_to_hf.py` | 上传已经转换好的本地数据集到 Hugging Face Dataset 仓库 | `LOCAL_DIR` 本地数据集目录；`HF_REPO_ID` 远端数据集仓库名；`PRIVATE` 是否上传为私有数据集 |
| `hugging_face/lerobot_data_info.py` | 检查 Hugging Face 上的 LeRobot 数据集信息 | `dataset_repo_id` 需要检查的数据集仓库名 |
| `hugging_face/push_model_to_hf.py` | 上传本地训练好的模型文件夹 | `LOCAL_MODEL_DIR` 本地模型目录；`HF_REPO_ID` 远端模型仓库名；`PATH_IN_REPO` 上传到仓库内的子目录 |
| `hugging_face/download_model_from_hf.py` | 从 Hugging Face 下载模型到本地 | `TARGET_REPO_ID` 远端模型仓库名；`BASE_PRETRAIN_DIR` 本地保存根目录 |

### 2. 原始数据格式
原始遥操数据默认保存位置：
```text
outputs/4_data_collect/quest_teleop/quest_teleop_SewNeedle-3Arms-v0
```
目录结构：
```text
metadata.json                         # 整个采集数据集的基础信息
episodes/
  episode_000000/
    info.json                         # 当前 episode 的成功标志、步数、相机路径等信息
    arrays.npz                        # 当前 episode 的数值轨迹
      joint_action                    # 关节动作，可直接用于 env.step
      pose_action                     # 可选，末端位姿动作，需要 IK 后才能用于关节控制
      observation_state               # 机器人状态观测
    videos/
      zed_cam_left.mp4                # 左 ZED 相机视频
      zed_cam_right.mp4               # 右 ZED 相机视频
      wrist_cam_left.mp4              # 左腕部相机视频
      wrist_cam_right.mp4             # 右腕部相机视频
      overhead_cam.mp4                # 顶部相机视频
      worms_eye_cam.mp4               # 底部相机视频
```

### 3. 转换后数据格式
转换后的本地 LeRobot/HF 数据集默认保存位置：
```text
outputs/5_hf_datasets/quest_teleop_sew_needle_3arms
```
目录结构：
```text
data/
  train-00000-of-00001.parquet        # 每帧的 state、action、timestamp、episode/frame/index 和视频帧引用

meta_data/
  info.json                           # 数据集基础信息，例如 fps、相机键、视频编码、episode 数量
  stats.safetensors                   # state、action、图像观测的 mean/std/min/max 统计量
  episode_data_index.safetensors      # 每条 episode 在全局帧序列中的起止索引

videos/
  observation.images.<camera>_episode_000000.mp4  # LeRobot 图像观测视频
```
转换后 `parquet` 中的核心字段：
```text
observation.state                     # 机器人状态
action                                # 训练动作，由 ACTION_KEY 指定来源
observation.images.<camera>           # 视频帧引用，包含 path + timestamp
episode_index                         # episode 编号
frame_index                           # episode 内帧编号
timestamp                             # 当前帧时间戳
next.done                             # 是否为 episode 最后一帧
index                                 # 全局帧编号
```
图像不会直接保存为数组，LeRobotDataset 训练时会根据 `path + timestamp` 从 mp4 中解码出对应图像帧。


## 🧾 小贴士
1. mujoco环境中`aloha_real.xml`比`aloha_sim.xml`多出以下两处聚光灯：
```
<light mode="targetbodycom" target="left_gripper_link" pos="-.5 .7 2.5" cutoff="55"/>
<light mode="targetbodycom" target="right_gripper_link" pos=".5 .7 2.5" cutoff="55"/>
```
2. 且两者的双目相机广角不一样，`aloha_real.xml`中`fovy="90"`，而`aloha_sim.xml`中为`fovy="66.21"`
```
<camera name="zed_cam_left" pos="0.03 0.00119254 -0.04325" euler="1.57079632679 0 3.14159265359" fovy="66.21" mode="fixed"/>
<camera name="zed_cam_right" pos="-0.03 0.00119254 -0.04325" euler="1.57079632679 0 3.14159265359" fovy="66.21" mode="fixed"/> 
```
3. 可以在评估eval.py代码中查看推理时间，一把来说一次会推理horizon步，这一次推理时间是最久的，后续只从推理的动作中取出并执行即可，所以推理时间会呈现 类似[31.86 ms、0.26 ms、0.31 ms、0.29 ms...]的分布,长度是实际执行的步数n_action_steps

4. 读取权重进行推理时建议使用policy=make_policy(...)实例化策略，相比于仅能读取裸模型参数的 DiffusionPolicy.from_pretrained()，make_policy 能够完整装载训练期参数（尤其是动作归一化高度依赖的 dataset_stats）。
