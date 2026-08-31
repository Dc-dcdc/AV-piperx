# PiperX 实机配置与安全测试指南

[English documentation: planned](#开源发布检查清单) | 简体中文

本指南说明如何在 Linux 上配置 PiperX、AGX 夹爪与 USB-CAN，并使用
AV-piper 提供的脚本，从只读通信检查逐步过渡到低速运动、夹爪和主从臂
测试。它面向首次实机部署和可复现实验，不替代制造商安全手册。

> [!WARNING]
> 本目录中的部分脚本能够使真实机械臂运动。软件限位、碰撞状态和电机
> 扭矩监控都不能替代风险评估、物理急停和现场监护。首次运行只能执行
> 只读测试。禁止在人、易碎物体或未经验证的夹具附近测试运动。

## 目录

- [适用范围](#适用范围)
- [测试矩阵](#测试矩阵)
- [安全模型](#安全模型)
- [快速开始：只读测试](#快速开始只读测试)
- [安装](#安装)
- [CAN 配置](#can-配置)
- [测试脚本](#测试脚本)
- [状态与反馈](#状态与反馈)
- [安全姿态](#安全姿态)
- [六关节运动](#六关节运动)
- [AGX 夹爪](#agx-夹爪)
- [示教器与主从臂](#示教器与主从臂)
- [碰撞与力反馈](#碰撞与力反馈)
- [故障处理](#故障处理)
- [日志与问题报告](#日志与问题报告)
- [日常操作清单](#日常操作清单)
- [开源发布检查清单](#开源发布检查清单)

## 适用范围

本文档覆盖：

- PiperX 六轴机械臂；
- AGX 夹爪；
- Linux SocketCAN 与 1 Mbps USB-CAN；
- `pyAgxArm` Python SDK；
- 单臂、Leader 主臂、Follower 从臂的分阶段测试；
- AV-piper 中 `real_robot/tests` 下的安全测试脚本。

不覆盖：

- 人机协作安全认证；
- 工业安全 PLC 或安全区域扫描器；
- 基于末端六维力传感器的力控；
- 未经验证的 `move_js`、MIT 或直接力矩控制部署；
- 具体工作站的碰撞模型和负载认证。

ZED 相机测试请直接查看：

```bash
python real_robot/tests/test_zed_camera.py --help
```

## 测试矩阵

以下组合已经完成基础通信和低速测试。它只表示该组合被验证过，不表示其他
版本一定不兼容。

| 组件 | 已测试版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS |
| Python | 3.10.19 |
| PiperX 固件 | `S-V1.9-0` |
| pyAgxArm | 1.0.0，commit `2255d88e1fab` |
| python-can | 4.6.1 |
| CAN 后端 | Linux SocketCAN |
| CAN 波特率 | 1 Mbps |
| 典型反馈频率 | 关节、法兰和状态约 200 Hz |

提交问题时应同时提供实际使用的版本，不能只写“最新版”。

## 安全模型

### 操作分级

| 等级 | 操作 | 现场要求 |
| --- | --- | --- |
| S0 | 安装、导入和 `--help` | 无实机运动 |
| S1 | CAN 与状态只读测试 | 机械臂可上电，但禁止运动 |
| S2 | 运动预检和 dry-run | 不发送真实目标 |
| S3 | 低速、小幅单关节运动 | 清空工作区、固定底座、急停可用 |
| S4 | 六轴、夹爪或主从跟随 | 完成前三级测试并有现场监护 |
| S5 | 策略闭环与高速控制 | 需独立安全层、看门狗和碰撞评估 |

### 立即停止条件

出现以下任一情况，应停止发送新目标并进行人工检查：

- 任一关节单独失能；
- 机械臂报告急停、碰撞、过流、堵转、制动器或通信异常；
- CAN 进入 `ERROR-PASSIVE` 或 `BUS-OFF`；
- 实际反馈向目标反方向运动；
- 目标误差长期不下降；
- 出现异常噪声、振动、发热或线缆拉扯；
- 主臂、相机或策略反馈超过看门狗时限；
- 示教器与 PC 同时拥有运动控制权。

不要在机械臂悬空承载时直接失能或断电。软件急停不能替代物理急停。

## 快速开始：只读测试

以下快速开始不会主动使能或移动机械臂。

### 1. 获取项目并创建环境

```bash
git clone <AV-piper-repository-url>
cd AV-piper
conda env create -f environment.yml
conda activate AV-piper
python -m pip install -e .
```

### 2. 安装 pyAgxArm

```bash
git clone https://github.com/agilexrobotics/pyAgxArm.git \
    third_party/pyAgxArm
export PYAGXARM_DIR="$(pwd)/third_party/pyAgxArm"
python -m pip install -e "$PYAGXARM_DIR"
```

为了复现上述测试矩阵，可在安装前检出对应 commit：

```bash
git -C "$PYAGXARM_DIR" checkout 2255d88e1fab
python -m pip install -e "$PYAGXARM_DIR"
```

### 3. 激活 CAN

```bash
bash "$PYAGXARM_DIR/pyAgxArm/scripts/ubuntu/can_activate.sh" \
    can_piperx 1000000
```

### 4. 检查 CAN 并运行只读测试

```bash
ip -details -statistics link show can_piperx

python real_robot/tests/test_piperx_connection.py
```

只有在终端显示 `PASS`，且 CAN 没有新增错误或丢包后，才进入后续章节。

## 安装

### 系统依赖

Ubuntu 需要 `git`、`can-utils` 和 `ethtool`：

```bash
sudo apt update
sudo apt install git can-utils ethtool
```

确认命令可用：

```bash
git --version
candump --help
ethtool --version
```

### Python 环境

推荐使用项目提供的 Conda 环境：

```bash
conda env create -f environment.yml
conda activate AV-piper
python -m pip install -e .
```

验证当前解释器，防止 SDK 安装到其他环境：

```bash
which python
python --version
python -m pip --version
```

### pyAgxArm

推荐“Git 克隆 + editable 安装”，因为 CAN 脚本、API 文档和示例也位于
源码仓库中：

```bash
git clone https://github.com/agilexrobotics/pyAgxArm.git \
    third_party/pyAgxArm
export PYAGXARM_DIR="$(pwd)/third_party/pyAgxArm"
python -m pip install -e "$PYAGXARM_DIR"
```

验证导入位置和版本：

```bash
python -c "import pyAgxArm; print(pyAgxArm.__file__)"
python -m pip show pyAgxArm python-can
```

更新 SDK 时不要静默改变实验环境。先记录旧 commit，再更新和重新测试：

```bash
git -C "$PYAGXARM_DIR" rev-parse HEAD
git -C "$PYAGXARM_DIR" pull --ff-only
python -m pip install -e "$PYAGXARM_DIR"
```

如果只需要安装包而不使用仓库内脚本，也可以直接安装：

```bash
python -m pip install \
    "git+https://github.com/agilexrobotics/pyAgxArm.git"
```

## CAN 配置

### 硬件检查

建议按以下顺序连接：

1. 固定机械臂底座并确认物理急停；
2. 检查控制器、夹爪、CAN-H、CAN-L 和参考地；
3. 检查终端电阻和设备要求的波特率；
4. 将 USB-CAN 接入稳定 USB 端口，避免无供电 Hub；
5. 最后给机械臂控制器上电。

不要在机械臂运动时插拔电机、夹爪或 CAN 总线连接器。

### 查找 USB-CAN

```bash
bash "$PYAGXARM_DIR/pyAgxArm/scripts/ubuntu/find_all_can_port.sh"
```

脚本会显示 Linux CAN 接口及其 USB 物理路径。更换 USB 端口后物理路径和
默认接口名可能变化。

### 激活单个 CAN 接口

```bash
bash "$PYAGXARM_DIR/pyAgxArm/scripts/ubuntu/can_activate.sh" \
    can_piperx 1000000
```

检查链路：

```bash
ip -details -statistics link show can_piperx
timeout 2 candump can_piperx
```

正常链路通常包含：

```text
UP, LOWER_UP
bitrate 1000000
can state ERROR-ACTIVE
```

`timeout 2 candump` 到时退出是预期行为。没有任何帧则应检查电源、接线、
波特率、终端电阻和接口名称。

### 多机械臂

多台机械臂推荐使用独立 USB-CAN 和唯一接口名，例如：

```text
can_left_leader
can_left_follower
can_right_leader
can_right_follower
```

多个未配置报文偏移的控制器不能直接并联到同一总线，否则相同 CAN ID
可能冲突。多设备时不要盲目使用只针对单个适配器的自动重命名流程，应先
记录 USB 物理路径并逐台验证。

### USB-CAN 重连

重新插拔后接口可能恢复为 `can0`，需要重新激活。部分适配器不支持
`restart-ms`，出现以下信息时不要反复配置：

```text
Device doesn't support restart from Bus Off
```

应先解决物理总线问题，再重新激活接口、重插 USB-CAN 或按制造商流程
重新上电。

## 测试脚本

所有命令从 AV-piper 仓库根目录执行。

| 脚本 | 用途 | 默认行为 | 真实运动安全门 |
| --- | --- | --- | --- |
| [`test_piperx_connection.py`](test_piperx_connection.py) | CAN、固件和反馈诊断 | 只读 | 无运动入口 |
| [`test_piperx_safe_pose.py`](test_piperx_safe_pose.py) | 进入安全初始姿态 | 仅预检 | `--allow-motion` + 确认词 |
| [`test_piperx_motion.py`](test_piperx_motion.py) | 六关节顺序小幅测试 | 仅预检 | `--allow-motion` + 确认词 |
| [`setup_piperx_gripper.py`](setup_piperx_gripper.py) | AGX 夹爪标定与初始化 | 只读预检 | `--allow-setup` + 交互确认 |
| [`test_zed_camera.py`](test_zed_camera.py) | ZED 取流和录像 | 相机测试 | 不控制机械臂 |

查看完整参数：

```bash
python real_robot/tests/test_piperx_connection.py --help
python real_robot/tests/test_piperx_safe_pose.py --help
python real_robot/tests/test_piperx_motion.py --help
python real_robot/tests/setup_piperx_gripper.py --help
```

## 状态与反馈

### 只读通信测试

```bash
python real_robot/tests/test_piperx_connection.py
```

典型通过条件：

- 固件识别成功；
- 关节、法兰和状态反馈频率满足脚本阈值；
- 数值有限且长度正确；
- 机械臂无异常状态；
- CAN 没有新增错误或丢包。

该测试不会调用 `enable()`、`move_j()` 或夹爪运动接口。`PASS` 只说明
通信正常，不代表目标姿态和运动路径安全。

### `ctrl_mode`

| 数值 | 模式 |
| --- | --- |
| 0 | Standby |
| 1 | CAN 指令控制 |
| 2 | 示教模式 |
| 3 | 以太网控制 |
| 4 | Wi-Fi 控制 |
| 5 | 遥控模式 |
| 6 | 主从联动示教输入 |
| 7 | 离线轨迹模式 |

### `teach_status`

| 数值 | 状态 |
| --- | --- |
| 0 | 未启用示教 |
| 1 | 开始拖动示教记录 |
| 2 | 结束示教记录 |
| 3 | 执行示教轨迹 |
| 4 | 暂停执行 |
| 5 | 继续执行 |
| 6 | 终止执行 |
| 7 | 移动到示教轨迹起点 |

### 运动与驱动状态

- `arm_status=NORMAL`：主状态正常；
- `REACH_TARGET_POS_SUCCESSFULLY`：控制器认为已到达目标；
- `REACH_TARGET_POS_FAILED`：尚未到达或无法到达目标；
- `collision_status=True`：对应驱动器报告碰撞保护；
- `driver_enable_status=False`：对应关节驱动器未使能。

脚本还会根据实际误差、连续稳定帧和使能状态二次判断，不能只看一个状态位。

## 安全姿态

该脚本仅用于当前位置靠近 SDK 限位、无法进行普通运动测试时。它不是每日
启动必须执行的回零程序。

### 预检

```bash
python real_robot/tests/test_piperx_safe_pose.py
```

默认目标为：

```text
J1: 保持反馈值
J2: 0.10 rad
J3: -0.08 rad
J4: -1.48 rad
J5: 保持反馈值
J6: 保持反馈值
```

### 真实运动

只有预检通过、六轴使能状态一致且完整路径无碰撞时运行：

```bash
python real_robot/tests/test_piperx_safe_pose.py \
    --allow-motion \
    --confirmation I_UNDERSTAND_PIPERX_WILL_MOVE_TO_SAFE_POSE
```

成功后默认保持使能，避免机械臂在悬空姿态下因失能下垂。

## 六关节运动

测试顺序为：

```text
保持共同初始姿态
→ J1小幅外移并返回
→ J2小幅外移并返回
→ ...
→ J6小幅外移并返回
```

每次只测试一个关节，不会同时施加六个测试偏移。

### 预检

```bash
python real_robot/tests/test_piperx_motion.py
```

### 真实运动

```bash
python real_robot/tests/test_piperx_motion.py \
    --allow-motion \
    --confirmation I_UNDERSTAND_PIPERX_WILL_MOVE
```

### 单关节预检

例如 J6 负向偏移 0.02 rad：

```bash
python real_robot/tests/test_piperx_motion.py \
    --joint-index 6 \
    --joint-offset-rad -0.02
```

该命令没有 `--allow-motion`，因此只执行预检。

### SDK 限位不是机械安全边界

`ArmModel.PIPER_X` 在已测试 SDK commit 中的预设为：

| 关节 | SDK 范围（rad） | 约合角度 |
| --- | ---: | ---: |
| J1 | `[-2.617994, 2.617994]` | `[-150°, 150°]` |
| J2 | `[0, 3.141593]` | `[0°, 180°]` |
| J3 | `[-2.967060, 0]` | `[-170°, 0°]` |
| J4 | `[-1.553344, 1.553344]` | 约 `[-89°, 89°]` |
| J5 | `[-1.553344, 1.553344]` | 约 `[-89°, 89°]` |
| J6 | `[-3.141593, 3.141593]` | `[-180°, 180°]` |

这些值必须与实际 SDK 版本核对。驱动器、装配零点、末端负载、线缆和夹具
都可能形成更严格的边界。

> [!NOTE]
> **单机现场记录，不是 PiperX 通用限位：**一台固件为 `S-V1.9-0` 的
> 设备曾在 J6 约 2.94 rad 继续正向运动时报告
> `REACH_TARGET_POS_FAILED`，随后 J6 单独失能。因此该工作站的测试方向
> 被改为远离该边界。其他设备必须独立标定和验证，不能照搬这一数值。

## AGX 夹爪

### 只读预检

```bash
python real_robot/tests/setup_piperx_gripper.py
```

检查内容包括反馈频率、宽度、力、使能、故障、有效行程和本地标定记录。

### 初始化

```bash
python real_robot/tests/setup_piperx_gripper.py --allow-setup
```

程序仍会要求操作者在当前终端输入确认词。确认夹爪内没有物体、手指和
线缆后，才允许低力分段运动。

### 零点标定

没有匹配记录时，程序会要求：

1. 确认夹爪失能；
2. 手动轻柔地完全闭合夹爪；
3. 输入现场确认词；
4. 将当前位置设置为零点；
5. 写入本地标定记录；
6. 低力分段张开并验证反馈。

只有更换夹爪、重装传动机构或确认零点错误时才重新标定：

```bash
python real_robot/tests/setup_piperx_gripper.py \
    --allow-setup --force-recalibration
```

不要把重复标定作为日常上电步骤，也不要通过持续增加夹持力解决未知卡滞。

## 示教器与主从臂

### 手持示教器

1. PC 只运行只读连接测试；
2. 在示教器上进入拖动模式；
3. 缓慢移动单个关节，检查反馈方向和连续性；
4. 退出拖动模式并确认机械臂稳定；
5. 验证物理急停；
6. PC 控制前完全退出示教记录和回放。

禁止示教器和 PC 同时发送运动目标。

### Leader–Follower

主从系统必须分阶段测试：

```text
两臂分别只读PASS
→ Leader只读反馈PASS
→ Follower独立运动PASS
→ 主从映射dry-run PASS
→ 单关节低速跟随
→ 六关节低缩放跟随
→ 夹爪跟随
```

Leader 第一阶段只读取：

```python
get_leader_joint_angles()
get_arm_status()
get_motor_states(joint_index)
```

不要在 Leader 首次检查时运行 `test_piperx_motion.py`。

Follower 目标不能直接等于 Leader 绝对角，应使用相对初始姿态映射：

\[
q_F^{cmd}=q_F^0+s\odot k\odot(q_L-q_L^0)
\]

其中 (s) 是逐关节方向，(k) 是缩放比例。镜像安装时各关节符号可能
不同。首次真实跟随建议只开放一个关节，并把缩放比例设为 0.1～0.2。

SDK 提供 `set_leader_mode()`、`set_follower_mode()`、
`move_leader_to_home()` 和 `move_leader_follower_to_home()`。这些接口会切换
控制模式或产生运动，不能用于首次只读检查。

## 碰撞与力反馈

### `move_j`

`move_j([j1, ..., j6])` 发送六轴绝对目标角，属于位置速度模式。控制器会
平滑运动，但不会理解场景中的障碍物，也不提供自动路径避障或末端力控。

即使只移动一个关节，也必须发送完整六轴目标。两个端点安全不代表中间的
关节空间路径安全。

### `move_js`

`move_js` 使用更直接的 JS/MIT 透传模式，没有 `move_j` 的平滑和轨迹规划。
它不能解决碰撞问题，目标跳变还可能引起冲击、振荡或失稳。首次部署禁止
用 `move_js` 替代 `move_j`。

### 可读取的反馈

pyAgxArm 提供：

- `get_motor_states(i)`：位置、速度、电流和估算电机扭矩；
- `get_driver_states(i)`：过流、堵转、碰撞和使能状态；
- `get_crash_protection_rating()`：逐关节碰撞保护等级。

电机扭矩不是末端六维力。它同时包含重力、惯性、摩擦和接触载荷，不能用
单一固定阈值直接判断所有姿态下的接触。

策略控制前应实现独立安全层：

```text
策略输出
→ 关节限位
→ 单步变化限幅
→ 速度/加速度限幅
→ 工作空间与路径碰撞检查
→ 电流/扭矩残差与collision_status监控
→ 反馈超时看门狗
→ move_j
```

## 故障处理

### 未收到固件信息

检查控制器电源、USB-CAN、接口 `UP` 状态、1 Mbps 波特率、CAN 接线、终端
电阻、`candump` 输出和是否有其他进程占用接口。

### CAN `ERROR-PASSIVE` 或 `BUS-OFF`

不要通过循环发命令或无限重启掩盖错误。检查 CAN-H/CAN-L、终端电阻、
波特率、供电、USB 稳定性和重复 CAN ID。修复物理问题后再重新激活接口。

### 六轴使能状态不一致

例如：

```text
[True, True, True, True, True, False]
```

表示至少一个关节已触发保护或失能。程序拒绝运动是正确行为。不要直接
重跑或循环强制使能。应检查机械边界、线缆和负载，按制造商流程清错、复位
或重新上电，再从只读测试重新开始。

### `REACH_TARGET_POS_FAILED`

可能原因包括实际机械边界、外物阻挡、静摩擦、错误控制模式、驱动失能或
目标稳定误差未达到阈值。不能简单通过延长超时解决；先确认反馈是否仍朝
目标方向运动。

### 程序运行但机械臂不动

看到以下输出表示只完成了预检：

```text
预检结果: PASS（未运动）
```

这属于默认安全行为。真实运动必须显式提供 `--allow-motion` 或
`--allow-setup`，并满足确认词、使能、控制模式和状态检查。

### J3/J4 略超 SDK 限位

断电下垂或零点差异可能让反馈略超 SDK 预设。由于 `move_j` 发送完整六轴
目标，即使只测试其他关节，软件限幅也可能让 J3/J4发生修正。修正量明显时
应先使用安全姿态预检，不能放宽阈值掩盖问题。

### 夹爪使能但不运动

检查控制模式、夹爪故障、标定记录、零点、有效行程和机械卡滞。不要通过
不断增加夹持力解决未知故障。

## 日志与问题报告

测试输出默认写入：

```text
outputs/6_real_robot_eval/<test_name>/<timestamp>/
```

常见文件：

- `summary.json`：配置、状态、检查项和结论；
- `samples.jsonl`：只读反馈；
- `trajectory.jsonl`：运动反馈；
- `calibration_record.json`：夹爪标定记录。

公开日志前应删除设备序列号、用户名、绝对路径、IP 地址和私有数据路径。

提交问题时请提供：

```text
操作系统：
Python版本：
pyAgxArm版本和commit：
python-can版本：
PiperX型号和固件：
CAN接口和状态：
执行命令：
是否允许真实运动：
期望行为：
实际行为：
是否触发急停、碰撞或关节失能：
脱敏后的summary.json：
```

不要只上传截图；文本日志和脱敏后的 JSON 更容易复现。

## 日常操作清单

### 启动

```text
固定底座并检查急停
→ 检查工作区、负载与线缆
→ 控制器上电
→ 连接并激活USB-CAN
→ 确认ERROR-ACTIVE和1 Mbps
→ 运行只读连接测试
→ 检查六轴使能、控制模式和错误码
→ 运行目标脚本预检
→ 现场确认后才允许真实运动
```

### 关机

1. 停止策略和所有运动发送进程；
2. 将机械臂移动到稳定的安全停放姿态；
3. 停止示教回放或主从跟随；
4. 根据负载和制造商流程决定何时失能；
5. 确认停止后关闭控制器电源；
6. 最后断开 USB-CAN 和其他外设。

### 从基础测试到策略部署

```text
CAN只读
→ 单臂反馈
→ 安全姿态
→ 单关节小幅运动
→ 六关节顺序运动
→ 夹爪
→ 多臂并行只读
→ 相机与机器人时间同步
→ 策略只推理dry-run
→ 低速闭环
```

不要从通信测试直接跳到多臂全速策略部署。

## 开源发布检查清单

当前文档可作为开源候选稿。正式发布前，仓库维护者还应完成：

- [ ] 将文件按 GitHub 惯例命名为 `README.md`；
- [ ] 在仓库根目录添加明确的 `LICENSE`；
- [ ] 添加 `CONTRIBUTING.md` 和 Issue 模板；
- [ ] 提供英文入口或英文版文档；
- [ ] 确认所有命令不包含个人绝对路径；
- [ ] 固定或记录经过测试的 pyAgxArm tag/commit；
- [ ] 对公开日志和图片进行隐私脱敏；
- [ ] 使用 Markdown link checker 检查相对链接；
- [ ] 修改脚本参数后同步更新本指南；
- [ ] 明确区分 AV-piper 代码许可与 pyAgxArm 自身许可。

欢迎通过 Issue 提交可复现的错误报告或文档改进。涉及机械臂真实运动的变更
应同时说明安全假设、失败处理和验证范围。

