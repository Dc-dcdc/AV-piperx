import os
from huggingface_hub import snapshot_download

def download_model_to_pretrain(repo_id: str, base_dir: str):
    """
    自动解析 Hugging Face 仓库 ID，在本地创建对应文件夹并下载模型。
    
    参数:
        repo_id (str): Hugging Face 上的目标仓库 ID (例如 "Dc-dc/model_name")
        base_dir (str): 本地保存的基础路径 (例如 "./pretrain")
    """
    # 1. 自动处理路径与文件夹
    # 提取模型名称并拼接目标路径
    model_name = repo_id.split("/")[-1]

    target_dir = os.path.join(base_dir, model_name)
    # target_dir = os.path.join(base_dir, model_name, "pretrained_model")

    
    # 安全创建文件夹
    os.makedirs(target_dir, exist_ok=True)
    print(f"📁 解析完毕！模型将存放在专属文件夹: {target_dir}")

    # 2. 配置网络环境
    os.environ["http_proxy"] = "http://127.0.0.1:7897"
    os.environ["https_proxy"] = "http://127.0.0.1:7897"
    print(f"🔄 正在从云端拉取 {model_name} 的完整文件...")

    # 3. 执行高速下载
    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        resume_download=True,  # 开启断点续传
    )

    print(f"✅ 下载大功告成！所有文件已妥善保存在: {local_path}")
    return local_path


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 在这里配置你的实际参数
    # 目标仓库
    TARGET_REPO_ID = "Dc-dc/pre_sim_insert_cylinder_3arms_zed_wrist_diffusion"
    
    # 本地文件夹，会在后面自动创建任务+相机名称
    BASE_PRETRAIN_DIR = "outputs/1_hugging_model"
    
    # 2. 调用函数执行下载
    download_model_to_pretrain(
        repo_id=TARGET_REPO_ID, 
        base_dir=BASE_PRETRAIN_DIR
    )