import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def inspect_lerobot_dataset():
    # ==========================================
    # 1. 配置参数
    # ==========================================
    # 替换成你的实际 repo_id 或本地路径lerobot/piperx_insert_cylinder_human
    dataset_repo_id = "iantc104/gv_sim_insert_cylinder_3arms"

    # 传入你的 resolved_delta_timestamps（如果之前定义了的话）
    delta_timestamps = None

    print(f"⏳ 正在加载数据集: {dataset_repo_id} ...")

    # ==========================================
    # 2. 初始化离线数据集
    # ==========================================
    offline_dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        delta_timestamps=delta_timestamps
    )

    # ==========================================
    # 3. 查看数据集的整体基础信息
    # ==========================================
    print("\n" + "="*50)
    print(" 📊 【1】数据集整体信息 (Dataset Info)")
    print("="*50)
    print(f"► 总帧数 (Total frames) : {len(offline_dataset)}")
    print(f"► 总片段数 (Episodes)   : {offline_dataset.num_episodes}")
    print(f"► 帧率 (FPS)            : {offline_dataset.fps}")
    if hasattr(offline_dataset, 'camera_keys'):
        print(f"► 相机视角 (Cameras)    : {offline_dataset.camera_keys}")

    # ==========================================
    # 4. 查看数据字典包含哪些键 (Keys) 以及注册信息 [已修复]
    # ==========================================
    print("\n" + "="*50)
    print(" 🔑 【2】数据集特征定义 (Features)")
    print("="*50)
    for key, feature in offline_dataset.features.items():
        print(f"● {key}:")

        # 安全地获取类型信息：尝试获取 dtype，如果没有则尝试获取 _type，再没有则打印类名
        f_type = getattr(feature, 'dtype', getattr(feature, '_type', type(feature).__name__))
        # 安全地获取形状：有些特殊对象没有 shape 属性
        f_shape = getattr(feature, 'shape', '动态形状/未指定')

        print(f"   ├─ 数据类型 (type): {f_type}")
        print(f"   └─ 基础形状 (shape): {f_shape}")

    # ==========================================
    # 5. 获取单条数据并查看实际加载的 Tensor 维度
    # ==========================================
    print("\n" + "="*50)
    print(" 📐 【3】单条样本的实际张量维度 (Tensor Shapes)")
    print("="*50)

    sample = offline_dataset[0]

    for key, data in sample.items():
        if isinstance(data, torch.Tensor):
            shape_str = str(list(data.shape))
            dtype_str = str(data.dtype).split('.')[-1]
            print(f"► {key: <30} | 维度: {shape_str: <20} | 类型: {dtype_str}")
        else:
            print(f"► {key: <30} | 值: {data} (Type: {type(data).__name__})")

    print("\n✅ 数据集检查完毕！")

if __name__ == "__main__":
    inspect_lerobot_dataset()
