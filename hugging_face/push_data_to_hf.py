#!/usr/bin/env python
"""将已经转换好的本地 LeRobot/HF 数据集上传到 Hugging Face。

该脚本不负责转换原始遥操数据。请先运行
``hugging_face/convert_data_to_hf.py`` 生成本地数据集目录，
然后再用本脚本上传到 Hugging Face 数据集仓库。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


# 初始化日志输出格式。
def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# 应用运行时代理等环境变量。
def apply_runtime_env(args: argparse.Namespace) -> None:
    if args.http_proxy:
        os.environ["http_proxy"] = args.http_proxy
    if args.https_proxy:
        os.environ["https_proxy"] = args.https_proxy


# 检查本地目录是否是已转换好的 LeRobot 数据集。
def validate_local_dataset_dir(local_dir: Path) -> None:
    required_paths = [
        local_dir / "data",
        local_dir / "meta_data" / "info.json",
        local_dir / "meta_data" / "stats.safetensors",
        local_dir / "meta_data" / "episode_data_index.safetensors",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"{local_dir} does not look like a converted LeRobot dataset. Missing:\n{missing_text}"
        )


# 上传本地数据集目录到 Hugging Face Hub。
def push_dataset(
    *,
    local_dir: str | Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    revision: str | None = None,
    commit_message: str = "Upload LeRobot collected dataset",
) -> None:
    local_dir = Path(local_dir)
    validate_local_dataset_dir(local_dir)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=commit_message,
    )
    logging.info("Pushed dataset to https://huggingface.co/datasets/%s", repo_id)


# 创建命令行参数解析器。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="上传已经转换好的 LeRobot/Hugging Face 本地数据集目录。"
    )
    parser.add_argument(
        "--local-dir",
        default="outputs/5_hf_datasets/quest_teleop_insert_cylinder_3arms",
        help="convert_data_to_hf.py 生成的本地 HF 数据集目录。",
    )
    parser.add_argument(
        "--repo-id",
        default="Dc-dc/quest_teleop_insert_cylinder_3arms",
        help="Hugging Face 数据集仓库名，格式为 用户名/数据集名。",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否创建/上传为私有数据集仓库。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token；None 表示使用 huggingface-cli login 保存的 token。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="上传目标分支或版本；None 表示默认 main。",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload LeRobot collected dataset",
        help="上传到 Hub 时显示的提交信息。",
    )
    parser.add_argument(
        "--http-proxy",
        default="",
        help="HTTP 代理地址；留空表示不修改当前环境变量。",
    )
    parser.add_argument(
        "--https-proxy",
        default="",
        help="HTTPS 代理地址；留空表示不修改当前环境变量。",
    )
    return parser


# 按命令行参数执行上传流程。
def run_from_args(args: argparse.Namespace) -> None:
    apply_runtime_env(args)
    push_dataset(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
        revision=args.revision,
        commit_message=args.commit_message,
    )


# 使用 Python 变量调用上传流程。
def push_local_dataset_to_hf(
    local_dir: str,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    revision: str | None = None,
    commit_message: str = "Upload LeRobot collected dataset",
    http_proxy: str = "",
    https_proxy: str = "",
) -> None:
    init_logging()
    args = argparse.Namespace(
        local_dir=local_dir,
        repo_id=repo_id,
        private=private,
        token=token,
        revision=revision,
        commit_message=commit_message,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
    )
    run_from_args(args)


# 命令行入口函数。
def main() -> None:
    init_logging()
    parser = build_arg_parser()
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        raise SystemExit

    # convert_data_to_hf.py 生成的本地 HF 数据集目录。
    LOCAL_DIR = "outputs/5_hf_datasets/quest_teleop_insert_cylinder_3arms_rgb_joint"

    # Hugging Face 数据集仓库，格式必须是 用户名/数据集名。
    HF_REPO_ID = "Dc-dc/quest_teleop_insert_cylinder_3arms_rgb_joint"

    # 是否上传为私有数据集。
    PRIVATE = False

    push_local_dataset_to_hf(
        local_dir=LOCAL_DIR,
        repo_id=HF_REPO_ID,
        private=PRIVATE,
    )
