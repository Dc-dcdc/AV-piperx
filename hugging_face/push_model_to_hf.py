#!/usr/bin/env python
"""Push a local LeRobot pretrained_model folder to Hugging Face Hub.

不需要把 token 写进代码。推荐先在终端执行一次：

    huggingface-cli login

之后 HfApi 会自动读取本地缓存的 token。
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def apply_network_env(
    *,
    http_proxy: str = "",
    https_proxy: str = "",
    clear_hf_endpoint: bool = True,
) -> None:
    """设置上传时的网络环境。"""
    if clear_hf_endpoint:
        os.environ.pop("HF_ENDPOINT", None)
    if http_proxy:
        os.environ["http_proxy"] = http_proxy
    if https_proxy:
        os.environ["https_proxy"] = https_proxy


def check_model_dir(local_dir: str | Path) -> Path:
    """检查待上传目录是否存在，并提醒目录内容。"""
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {local_dir}")
    if not local_dir.is_dir():
        raise NotADirectoryError(f"不是文件夹: {local_dir}")

    expected_files = ["config.json", "config.yaml"]
    missing = [name for name in expected_files if not (local_dir / name).exists()]
    if missing:
        print(f"提示: {local_dir} 中没有找到 {missing}，请确认这是 pretrained_model 目录。")
    return local_dir


def push_model_folder_to_hf(
    local_dir: str | Path,
    repo_id: str,
    *,
    path_in_repo: str = "pretrained_model",
    commit_message: str = "Upload model",
    private: bool = False,
    token: str | None = None,
    http_proxy: str = "",
    https_proxy: str = "",
    clear_hf_endpoint: bool = True,
) -> None:
    """上传本地模型文件夹到 Hugging Face Hub。

    token=None 时使用 huggingface-cli login 保存的 token。
    """
    apply_network_env(
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        clear_hf_endpoint=clear_hf_endpoint,
    )
    local_dir = check_model_dir(local_dir)
    api = HfApi(token=token) # 创建 API 实例，自动使用登录的 token（如果 token=None）

    print(f"准备上传本地模型目录: {local_dir}")
    print(f"目标仓库: https://huggingface.co/{repo_id}")
    print(f"仓库内路径: {path_in_repo}")
    # 创建仓库（如果已存在且 exist_ok=True 则不会报错）
    api.create_repo(
        repo_id=repo_id,    # 仓库 ID，格式通常是 "用户名/仓库名"
        repo_type="model",  # 仓库类型，这里是模型仓库
        private=private,    # 是否创建为私有仓库
        exist_ok=True,      # 如果仓库已存在则继续上传，不会报错
    )
    # 上传模型文件夹
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,       # 仓库内的目标路径
        commit_message=commit_message,   # 提交信息
    )

    print("模型上传成功。")


if __name__ == "__main__":
    # 你的本地模型文件夹路径；通常末尾是 pretrained_model。
    LOCAL_MODEL_DIR = "outputs/2_pretrain/train/2026-05-18/21-39-48_InsertCylinder-3Arms-v0_pre_zed_wrist_act/checkpoints/032000_loss=0.0729_sr=0.0_ar=-113.35/pretrained_model"

    # 你的 Hugging Face 模型仓库，格式为 用户名/仓库名。
    HF_REPO_ID = "Dc-dc/pre_sim_insert_cylinder_3arms_zed_wrist_act"

    # 上传到仓库内的子目录；如果想上传到仓库根目录，改成 "".
    PATH_IN_REPO = "pretrained_model"

    # 是否创建/上传为私有仓库。
    PRIVATE = False

    # 不要把 token 写进代码；None 表示使用 huggingface-cli login 保存的 token。
    HF_TOKEN = None

    # 如需代理，在这里填写；不需要就保持空字符串。
    HTTP_PROXY = ""
    HTTPS_PROXY = ""

    push_model_folder_to_hf(
        LOCAL_MODEL_DIR,
        HF_REPO_ID,
        path_in_repo=PATH_IN_REPO,
        private=PRIVATE,
        token=HF_TOKEN,
        http_proxy=HTTP_PROXY,
        https_proxy=HTTPS_PROXY,
    )
