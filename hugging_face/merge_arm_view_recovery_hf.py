#!/usr/bin/env python
"""合并同一任务的 Arm/View/Mixed 恢复 LeRobot 数据集。

合并结果只保留一份原始专家轨迹，并保留各输入数据集中的全部恢复分支：

    原始专家 episode（去重）
      + Arm recovery episode
      + View recovery episode
      + Mixed recovery episode（可选）

脚本依赖源 HF 数据集 ``meta_data/info.json`` 中的 ``source_raw_dir``，
通过 raw episode 的 ``info.json`` 恢复 ``source_episode``、``variant_index``
和 ``is_augmented``，避免依靠转换后的连续 episode_index 猜测数据类型。

输出仍是 train_pretrain.py 可直接读取的单个本地 LeRobot 数据集。视频默认
使用硬链接，既不重复占用磁盘空间，又不会像符号链接一样依赖输出目录位置；
硬链接不可用时会自动退回普通复制。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
root_path = str(ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)


RAW_EPISODE_PATTERN = re.compile(
    r"^episode_(?P<source>\d{6,})(?:_aug_(?P<variant>\d{2,}))?$"
)
NUMERIC_STAT_KEYS = (
    "observation.state",
    "action",
    "episode_index",
    "frame_index",
    "timestamp",
    "next.done",
    "index",
)


@dataclass(frozen=True)
class DatasetEpisode:
    """源 HF episode 与对应 raw episode 元数据。"""

    dataset_role: str
    dataset_dir: Path
    old_episode_index: int
    start: int
    end: int
    raw_episode_dir: Path
    raw_episode_name: str
    source_episode: int
    variant_index: int
    is_augmented: bool
    raw_info: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return self.end - self.start

    @property
    def recovery_type(self) -> str:
        if not self.is_augmented:
            return "original"
        return f"{self.dataset_role}_recovery"


@dataclass
class DatasetSource:
    """一个源 HF 数据集及其 episode 映射。"""

    role: str
    dataset_dir: Path
    raw_dir: Path
    info: dict[str, Any]
    table: pa.Table
    episodes: list[DatasetEpisode]
    stats: dict[str, torch.Tensor]


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def parse_raw_episode_name(name: str) -> tuple[int, int]:
    match = RAW_EPISODE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            "raw episode目录必须形如episode_000003或episode_000003_aug_00，"
            f"当前为{name!r}。"
        )
    variant = match.group("variant")
    return int(match.group("source")), (-1 if variant is None else int(variant))


def list_raw_episode_dirs(raw_dir: Path) -> list[Path]:
    episodes_dir = raw_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"找不到raw episodes目录: {episodes_dir}")

    identities: dict[tuple[int, int], Path] = {}
    for path in episodes_dir.glob("episode_*"):
        if not path.is_dir() or path.name.endswith(".tmp"):
            continue
        if not (path / "arrays.npz").is_file() or not (path / "info.json").is_file():
            continue
        identity = parse_raw_episode_name(path.name)
        if identity in identities:
            raise RuntimeError(
                f"raw数据存在重复source/variant: {identities[identity]} 与 {path}"
            )
        identities[identity] = path

    if not identities:
        raise FileNotFoundError(f"{episodes_dir} 中没有可用episode。")

    # 与 convert_data_to_hf.py 保持完全相同的排序：每个source先原始、后增强。
    return [
        path
        for _, path in sorted(
            identities.items(),
            key=lambda item: (item[0][0], item[0][1] + 1),
        )
    ]


def raw_episode_dirs_from_manifest(
    *,
    dataset_dir: Path,
    raw_dir: Path,
    info: dict[str, Any],
    expected_episode_count: int,
) -> list[Path] | None:
    """按抽样清单恢复HF episode到raw episode的一一映射。

    完整转换的数据集没有 ``episode_manifest``，仍由
    :func:`list_raw_episode_dirs` 按raw目录排序映射。经过子采样的数据集则不能
    再假设raw/HF数量相同，必须以清单中的episode顺序为准。
    """

    manifest_value = info.get("episode_manifest")
    if not manifest_value:
        return None
    manifest_path = Path(str(manifest_value)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = dataset_dir / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_relative_to(dataset_dir):
        raise ValueError(
            "episode_manifest必须位于当前HF数据集目录内: "
            f"dataset={dataset_dir}, manifest={manifest_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到episode_manifest: {manifest_path}")

    manifest = read_json(manifest_path)
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise ValueError(f"{manifest_path}缺少episodes列表。")
    if len(records) != int(expected_episode_count):
        raise ValueError(
            f"episode_manifest与HF索引数量不一致: "
            f"manifest={len(records)}, hf={expected_episode_count}"
        )

    raw_episodes_root = (raw_dir / "episodes").resolve()
    raw_episode_dirs: list[Path] = []
    identities: set[tuple[int, int]] = set()
    for hf_episode_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"episode_manifest记录[{hf_episode_index}]必须是对象。"
            )
        declared_index = record.get("new_episode_index")
        if declared_index is None or int(declared_index) != hf_episode_index:
            raise ValueError(
                f"episode_manifest记录[{hf_episode_index}]的new_episode_index="
                f"{declared_index!r}，必须与HF episode顺序一致。"
            )

        raw_episode_value = record.get("raw_episode_dir")
        raw_episode_name = record.get("raw_episode_name")
        if raw_episode_value:
            raw_episode_dir = Path(str(raw_episode_value)).expanduser()
            if not raw_episode_dir.is_absolute():
                raw_episode_dir = raw_episodes_root / raw_episode_dir
        elif raw_episode_name:
            raw_episode_dir = raw_episodes_root / str(raw_episode_name)
        else:
            raise ValueError(
                f"episode_manifest记录[{hf_episode_index}]缺少"
                "raw_episode_dir/raw_episode_name。"
            )
        raw_episode_dir = raw_episode_dir.resolve()
        if not raw_episode_dir.is_relative_to(raw_episodes_root):
            raise ValueError(
                f"episode_manifest记录[{hf_episode_index}]指向source_raw_dir之外: "
                f"{raw_episode_dir}"
            )
        if raw_episode_name is not None and raw_episode_dir.name != str(raw_episode_name):
            raise ValueError(
                f"episode_manifest记录[{hf_episode_index}]的raw_episode_name="
                f"{raw_episode_name!r}与路径名{raw_episode_dir.name!r}不一致。"
            )
        if not (raw_episode_dir / "arrays.npz").is_file() or not (
            raw_episode_dir / "info.json"
        ).is_file():
            raise FileNotFoundError(
                f"episode_manifest引用的raw episode不完整: {raw_episode_dir}"
            )

        identity = parse_raw_episode_name(raw_episode_dir.name)
        if identity in identities:
            raise ValueError(
                "episode_manifest包含重复raw source/variant: "
                f"source={identity[0]}, variant={identity[1]}"
            )
        identities.add(identity)
        declared_source = record.get("source_episode")
        declared_variant = record.get("variant_index")
        if declared_source is not None and int(declared_source) != identity[0]:
            raise ValueError(
                f"episode_manifest记录[{hf_episode_index}]的source_episode="
                f"{declared_source}与raw目录{identity[0]}不一致。"
            )
        if declared_variant is not None and int(declared_variant) != identity[1]:
            raise ValueError(
                f"episode_manifest记录[{hf_episode_index}]的variant_index="
                f"{declared_variant}与raw目录{identity[1]}不一致。"
            )
        raw_episode_dirs.append(raw_episode_dir)
    return raw_episode_dirs


def read_parquet_dataset(dataset_dir: Path) -> pa.Table:
    parquet_paths = sorted((dataset_dir / "data").glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"找不到Parquet文件: {dataset_dir / 'data'}")

    tables = [pq.read_table(path) for path in parquet_paths]
    base_schema = tables[0].schema
    for path, table in zip(parquet_paths[1:], tables[1:], strict=True):
        if not table.schema.equals(base_schema, check_metadata=False):
            raise ValueError(f"同一数据集内Parquet schema不一致: {path}")
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def resolve_raw_dir(dataset_dir: Path, info: dict[str, Any]) -> Path:
    raw_value = info.get("source_raw_dir")
    if not raw_value:
        raise ValueError(
            f"{dataset_dir / 'meta_data/info.json'} 缺少source_raw_dir，"
            "无法可靠区分原始与恢复episode。"
        )
    raw_dir = resolve_path(str(raw_value))
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"source_raw_dir不存在: {raw_dir}。合并时需要保留原始raw元数据。"
        )
    return raw_dir


def load_dataset_source(role: str, dataset_dir_value: str | Path) -> DatasetSource:
    dataset_dir = resolve_path(dataset_dir_value)
    info_path = dataset_dir / "meta_data" / "info.json"
    stats_path = dataset_dir / "meta_data" / "stats.safetensors"
    episode_index_path = dataset_dir / "meta_data" / "episode_data_index.safetensors"
    for required_path in (info_path, stats_path, episode_index_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"数据集缺少文件: {required_path}")

    info = read_json(info_path)
    raw_dir = resolve_raw_dir(dataset_dir, info)
    table = read_parquet_dataset(dataset_dir)
    episode_data_index = load_file(episode_index_path)
    starts = episode_data_index.get("from")
    ends = episode_data_index.get("to")
    if starts is None or ends is None or len(starts) != len(ends):
        raise ValueError(f"非法episode_data_index: {episode_index_path}")
    raw_episode_dirs = raw_episode_dirs_from_manifest(
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        info=info,
        expected_episode_count=len(starts),
    )
    if raw_episode_dirs is None:
        raw_episode_dirs = list_raw_episode_dirs(raw_dir)
    if len(raw_episode_dirs) != len(starts):
        raise ValueError(
            f"{role}数据映射数量不一致: raw={len(raw_episode_dirs)}, "
            f"hf={len(starts)}。完整转换数据应与source_raw_dir数量一致；"
            "子采样数据必须在info.json中提供有效的episode_manifest。"
        )
    if int(info.get("total_episodes", len(starts))) != len(starts):
        raise ValueError(
            f"{role} info.total_episodes与索引不一致: "
            f"{info.get('total_episodes')} != {len(starts)}"
        )

    episodes: list[DatasetEpisode] = []
    previous_end = 0
    for old_episode_index, (raw_episode_dir, start_tensor, end_tensor) in enumerate(
        zip(raw_episode_dirs, starts, ends, strict=True)
    ):
        start = int(start_tensor.item())
        end = int(end_tensor.item())
        if start != previous_end or end <= start or end > table.num_rows:
            raise ValueError(
                f"{role} episode={old_episode_index}范围非法或不连续: "
                f"[{start}, {end}), previous_end={previous_end}, rows={table.num_rows}"
            )
        previous_end = end

        raw_info = read_json(raw_episode_dir / "info.json")
        parsed_source, parsed_variant = parse_raw_episode_name(raw_episode_dir.name)
        source_episode = int(raw_info.get("source_episode", parsed_source))
        variant_index = int(raw_info.get("variant_index", parsed_variant))
        is_augmented = bool(raw_info.get("is_augmented", parsed_variant >= 0))
        if source_episode != parsed_source:
            raise ValueError(
                f"{raw_episode_dir} 的source_episode={source_episode}与目录名{parsed_source}不一致。"
            )
        if variant_index != parsed_variant:
            raise ValueError(
                f"{raw_episode_dir} 的variant_index={variant_index}"
                f"与目录名{parsed_variant}不一致。"
            )
        if is_augmented != (parsed_variant >= 0):
            raise ValueError(
                f"{raw_episode_dir} 的is_augmented与目录名不一致。"
            )

        episode_values = table.column("episode_index").slice(start, end - start).to_pylist()
        if any(int(value) != old_episode_index for value in episode_values):
            raise ValueError(
                f"{role} Parquet episode_index与episode_data_index不一致: {old_episode_index}"
            )

        episodes.append(
            DatasetEpisode(
                dataset_role=role,
                dataset_dir=dataset_dir,
                old_episode_index=old_episode_index,
                start=start,
                end=end,
                raw_episode_dir=raw_episode_dir,
                raw_episode_name=raw_episode_dir.name,
                source_episode=source_episode,
                variant_index=variant_index,
                is_augmented=is_augmented,
                raw_info=raw_info,
            )
        )

    if previous_end != table.num_rows:
        raise ValueError(
            f"{role} episode索引未覆盖全部Parquet行: {previous_end} != {table.num_rows}"
        )

    return DatasetSource(
        role=role,
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        info=info,
        table=table,
        episodes=episodes,
        stats=load_file(stats_path),
    )


def validate_source_compatibility(sources: dict[str, DatasetSource]) -> None:
    """确认所有恢复数据集具有相同schema和专家episode集合。

    各角色的恢复episode数量可以不同；合并时会保留每个输入中的
    全部恢复分支，只对重复的原始专家episode去重。
    """

    if len(sources) < 2:
        raise ValueError("至少需要两个恢复HF数据集。")
    dataset_dirs = [source.dataset_dir for source in sources.values()]
    if len(set(dataset_dirs)) != len(dataset_dirs):
        raise ValueError("Arm/View/Mixed必须指向彼此不同的HF数据集目录。")
    reference_role, reference = next(iter(sources.items()))
    reference_original_ids = {
        episode.source_episode
        for episode in reference.episodes
        if not episode.is_augmented
    }
    for role, source in list(sources.items())[1:]:
        for key in ("codebase_version", "fps", "video", "camera_keys"):
            if reference.info.get(key) != source.info.get(key):
                raise ValueError(
                    f"{reference_role}/{role}数据集的{key}不一致: "
                    f"{reference.info.get(key)!r} != {source.info.get(key)!r}"
                )
        if not reference.table.schema.equals(source.table.schema, check_metadata=False):
            raise ValueError(
                f"{reference_role}/{role} Parquet schema不一致，不能合并。"
            )
        original_ids = {
            episode.source_episode
            for episode in source.episodes
            if not episode.is_augmented
        }
        if reference_original_ids != original_ids:
            only_reference = sorted(reference_original_ids - original_ids)
            only_current = sorted(original_ids - reference_original_ids)
            raise ValueError(
                f"{reference_role}/{role}原始专家episode集合不一致。"
                f" only_{reference_role}={only_reference[:20]}, "
                f"only_{role}={only_current[:20]}"
            )


def episode_lookup(source: DatasetSource) -> dict[tuple[int, int], DatasetEpisode]:
    result: dict[tuple[int, int], DatasetEpisode] = {}
    for episode in source.episodes:
        key = (episode.source_episode, episode.variant_index)
        if key in result:
            raise ValueError(f"{source.role}出现重复episode身份: {key}")
        result[key] = episode
    return result


def verify_original_episode_equality(
    sources: dict[str, DatasetSource], original_source: str
) -> None:
    """确认所有将被去掉的专家轨迹与保留版本数值完全相同。"""

    if original_source not in sources:
        raise ValueError(
            f"original_source={original_source!r}没有对应输入数据集。"
        )
    clean_source = sources[original_source]
    clean_lookup = episode_lookup(clean_source)
    compare_columns = [
        "observation.state",
        "action",
        "frame_index",
        "timestamp",
        "next.done",
    ]
    original_ids = sorted(
        episode.source_episode
        for episode in clean_source.episodes
        if not episode.is_augmented
    )
    comparison_count = 0
    for role, source in sources.items():
        if role == original_source:
            continue
        lookup = episode_lookup(source)
        for source_episode in original_ids:
            clean_episode = clean_lookup[(source_episode, -1)]
            compared_episode = lookup[(source_episode, -1)]
            clean_table = clean_source.table.slice(
                clean_episode.start, clean_episode.frame_count
            ).select(compare_columns)
            compared_table = source.table.slice(
                compared_episode.start, compared_episode.frame_count
            ).select(compare_columns)
            if not clean_table.equals(compared_table, check_metadata=False):
                raise ValueError(
                    f"发现{original_source}/{role}中的原始专家轨迹不相同，"
                    "已停止自动去重: "
                    f"source_episode={source_episode}。可先检查raw数据来源。"
                )
            comparison_count += 1
    logging.info(
        "已验证 %d 组重复原始专家轨迹数值完全一致。", comparison_count
    )


def choose_episodes(
    sources: dict[str, DatasetSource],
    original_source: str,
) -> list[DatasetEpisode]:
    if original_source not in sources:
        raise ValueError(
            f"original_source={original_source!r}没有对应输入数据集。"
        )
    clean_source = sources[original_source]
    originals = {
        episode.source_episode: episode
        for episode in clean_source.episodes
        if not episode.is_augmented
    }
    augmented_by_role: dict[str, dict[int, list[DatasetEpisode]]] = {
        role: {} for role in sources
    }
    for role, source in sources.items():
        target = augmented_by_role[role]
        for episode in source.episodes:
            if episode.is_augmented:
                target.setdefault(episode.source_episode, []).append(episode)

    selected: list[DatasetEpisode] = []
    for source_episode in sorted(originals):
        selected.append(originals[source_episode])
        for role in sources:
            selected.extend(
                sorted(
                    augmented_by_role[role].get(source_episode, []),
                    key=lambda episode: episode.variant_index,
                )
            )

    selected_augmented = {id(episode) for episode in selected if episode.is_augmented}
    all_augmented = {
        id(episode)
        for source in sources.values()
        for episode in source.episodes
        if episode.is_augmented
    }
    if selected_augmented != all_augmented:
        missing = len(all_augmented - selected_augmented)
        raise ValueError(
            f"有{missing}条恢复episode找不到对应原始专家episode，不能静默丢弃。"
        )
    return selected


def resolve_video_path(dataset_dir: Path, stored_path: str) -> Path:
    path = Path(stored_path)
    if not path.is_absolute():
        path = dataset_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parquet引用的视频不存在: {path}")
    return path


def materialize_video(source: Path, destination: Path, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"目标视频重复: {destination}")

    if link_mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if link_mode == "symlink":
        relative_source = os.path.relpath(source, start=destination.parent)
        destination.symlink_to(relative_source)
        return "symlink"
    if link_mode != "hardlink":
        raise ValueError(f"不支持的link_mode: {link_mode}")

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as error:
        logging.warning(
            "视频硬链接失败，将退回复制: source=%s, destination=%s, error=%s",
            source,
            destination,
            error,
        )
        shutil.copy2(source, destination)
        return "copy_fallback"


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise KeyError(f"Parquet缺少字段: {name}")
    return table.set_column(column_index, table.schema.field(column_index), values)


def rewrite_episode_table(
    source: DatasetSource,
    episode: DatasetEpisode,
    new_episode_index: int,
    global_index: int,
    camera_keys: list[str],
    staging_dir: Path,
    link_mode: str,
) -> tuple[pa.Table, dict[str, str], dict[str, str]]:
    table = source.table.slice(episode.start, episode.frame_count)
    episode_type = table.schema.field("episode_index").type
    index_type = table.schema.field("index").type
    table = replace_column(
        table,
        "episode_index",
        pa.array([new_episode_index] * episode.frame_count, type=episode_type),
    )
    table = replace_column(
        table,
        "index",
        pa.array(
            np.arange(global_index, global_index + episode.frame_count, dtype=np.int64),
            type=index_type,
        ),
    )

    source_video_paths: dict[str, str] = {}
    output_video_paths: dict[str, str] = {}
    for camera_key in camera_keys:
        camera_column = table.column(camera_key).combine_chunks()
        stored_paths = {
            value["path"]
            for value in camera_column.to_pylist()
            if value is not None and value.get("path")
        }
        if len(stored_paths) != 1:
            raise ValueError(
                f"episode={episode.old_episode_index} camera={camera_key}应只引用一个视频，"
                f"当前为{sorted(stored_paths)}"
            )
        stored_path = next(iter(stored_paths))
        source_video = resolve_video_path(source.dataset_dir, stored_path)
        output_relative = Path("videos") / f"{camera_key}_episode_{new_episode_index:06d}.mp4"
        output_video = staging_dir / output_relative
        actual_mode = materialize_video(source_video, output_video, link_mode)

        timestamp_array = camera_column.field("timestamp")
        new_camera_column = pa.StructArray.from_arrays(
            [
                pa.array([output_relative.as_posix()] * episode.frame_count, type=pa.string()),
                timestamp_array,
            ],
            names=["path", "timestamp"],
        )
        table = replace_column(table, camera_key, new_camera_column)
        source_video_paths[camera_key] = str(source_video)
        output_video_paths[camera_key] = output_relative.as_posix()
        output_video_paths[f"{camera_key}__materialization"] = actual_mode

    return table, source_video_paths, output_video_paths


def fixed_list_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if not pa.types.is_fixed_size_list(array.type):
        raise TypeError(f"期望fixed_size_list，当前为{array.type}")
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(array), array.type.list_size)


def vector_stats(values: np.ndarray) -> dict[str, torch.Tensor]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "mean": torch.from_numpy(array.mean(axis=0).astype(np.float32)),
        "std": torch.from_numpy(np.maximum(array.std(axis=0), 1e-6).astype(np.float32)),
        "min": torch.from_numpy(array.min(axis=0).astype(np.float32)),
        "max": torch.from_numpy(array.max(axis=0).astype(np.float32)),
    }


def build_merged_stats(
    table: pa.Table,
    camera_keys: list[str],
    sources: dict[str, DatasetSource],
) -> dict[str, torch.Tensor]:
    flattened: dict[str, torch.Tensor] = {}
    for key in NUMERIC_STAT_KEYS:
        if key in {"observation.state", "action"}:
            values = fixed_list_to_numpy(table.column(key))
        else:
            values = np.asarray(
                table.column(key).combine_chunks().to_numpy(zero_copy_only=False),
                dtype=np.float32,
            )
        for stat_name, tensor in vector_stats(values).items():
            flattened[f"{key}/{stat_name}"] = tensor

    # 当前转换流程在没有jpg帧时使用固定图像统计。只有所有来源统计一致时才可无损继承；
    # 若未来源数据改为真实图像统计，应先扩展此脚本为从合并视频重新抽帧统计。
    reference_role, reference = next(iter(sources.items()))
    for camera_key in camera_keys:
        for stat_name in ("mean", "std", "min", "max"):
            key = f"{camera_key}/{stat_name}"
            if key not in reference.stats:
                raise KeyError(f"{reference_role}源stats缺少图像统计: {key}")
            for role, source in sources.items():
                if key not in source.stats:
                    raise KeyError(f"{role}源stats缺少图像统计: {key}")
                if not torch.equal(reference.stats[key], source.stats[key]):
                    raise ValueError(
                        f"{reference_role}/{role}图像统计不一致: {key}。"
                        "为了避免写入不准确统计，请先统一图像统计覆盖，"
                        "或扩展脚本从视频重新计算。"
                    )
            flattened[key] = reference.stats[key].clone()
    return flattened


def write_support_files(
    staging_dir: Path,
    sources: dict[str, DatasetSource],
    original_source: str,
    total_episodes: int,
    total_frames: int,
    composition: dict[str, dict[str, int]],
) -> None:
    info = dict(sources[original_source].info)
    info.pop("source_raw_dir", None)
    has_mixed = "mixed" in sources
    merge_type = (
        "deduplicated_arm_view_mixed_recovery_v1"
        if has_mixed
        else "deduplicated_arm_view_recovery_v1"
    )
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "source_hf_dirs": [
                str(source.dataset_dir) for source in sources.values()
            ],
            "source_raw_dirs": [str(source.raw_dir) for source in sources.values()],
            "source_hf_dirs_by_role": {
                role: str(source.dataset_dir) for role, source in sources.items()
            },
            "source_raw_dirs_by_role": {
                role: str(source.raw_dir) for role, source in sources.items()
            },
            "merge_type": merge_type,
            "composition": composition,
            "episode_manifest": "recovery_manifest.json",
        }
    )
    write_json(staging_dir / "meta_data" / "info.json", info)

    readme = (
        "---\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "- recovery\n"
        "---\n"
        f"# Deduplicated {'Arm/View/Mixed' if has_mixed else 'Arm/View'} recovery dataset\n\n"
        "This dataset contains one copy of each clean expert episode, all Arm recovery "
        "branches, all View recovery branches"
        + (", and all Mixed recovery branches" if has_mixed else "")
        + ". See `recovery_manifest.json` for "
        "episode provenance.\n"
    )
    (staging_dir / "README.md").write_text(readme, encoding="utf-8")
    (staging_dir / ".gitattributes").write_text(
        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n"
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )


def merge_datasets(args: argparse.Namespace) -> Path | None:
    arm = load_dataset_source("arm", args.arm_dir)
    view = load_dataset_source("view", args.view_dir)
    sources = {"arm": arm, "view": view}
    # argparse 的可选路径默认可能是空字符串；空字符串表示未启用 Mixed，
    # 不能把它解析成项目根目录后再尝试加载数据集。
    if args.mixed_dir is not None and str(args.mixed_dir).strip():
        sources["mixed"] = load_dataset_source("mixed", args.mixed_dir)
    validate_source_compatibility(sources)
    if args.verify_originals:
        verify_original_episode_equality(sources, args.original_source)

    selected = choose_episodes(sources, args.original_source)
    composition: dict[str, dict[str, int]] = {
        "original": {"episodes": 0, "frames": 0},
        "arm_recovery": {"episodes": 0, "frames": 0},
        "view_recovery": {"episodes": 0, "frames": 0},
    }
    if "mixed" in sources:
        composition["mixed_recovery"] = {"episodes": 0, "frames": 0}
    for episode in selected:
        bucket = composition[episode.recovery_type]
        bucket["episodes"] += 1
        bucket["frames"] += episode.frame_count

    total_frames = sum(episode.frame_count for episode in selected)
    logging.info(
        "合并计划: episodes=%d, frames=%d, composition=%s",
        len(selected),
        total_frames,
        composition,
    )
    if args.dry_run:
        logging.info("dry-run完成，未写入任何文件。")
        return None

    output_dir_value = args.output_dir
    if output_dir_value is None:
        suffix = (
            "arm_view_mixed_random_recovery_3+0"
            if "mixed" in sources
            else "arm_view_random_recovery_3+0"
        )
        output_dir_value = (
            "outputs/5_hf_datasets/InsertCylinder-3Arms/"
            f"quest_teleop_InsertCylinder-3Arms-v0_rgb_{suffix}"
        )
    output_dir = resolve_path(output_dir_value)
    if output_dir == ROOT:
        raise ValueError("output-dir不能是项目根目录。")
    for source_dir in (source.dataset_dir for source in sources.values()):
        if output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir):
            raise ValueError(
                "output-dir不能与源数据集相同，也不能是源数据集的父目录或子目录: "
                f"output={output_dir}, source={source_dir}"
            )
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出目录已存在: {output_dir}。确认覆盖时请传入--overwrite。"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    (staging_dir / "data").mkdir(parents=True, exist_ok=True)
    (staging_dir / "meta_data").mkdir(parents=True, exist_ok=True)
    (staging_dir / "videos").mkdir(parents=True, exist_ok=True)

    episode_tables: list[pa.Table] = []
    episode_from: list[int] = []
    episode_to: list[int] = []
    manifest_episodes: list[dict[str, Any]] = []
    global_index = 0
    try:
        for new_episode_index, episode in enumerate(selected):
            source = sources[episode.dataset_role]
            episode_from.append(global_index)
            rewritten, source_videos, output_videos = rewrite_episode_table(
                source=source,
                episode=episode,
                new_episode_index=new_episode_index,
                global_index=global_index,
                camera_keys=list(sources[args.original_source].info["camera_keys"]),
                staging_dir=staging_dir,
                link_mode=args.link_mode,
            )
            episode_tables.append(rewritten)
            global_index += episode.frame_count
            episode_to.append(global_index)
            manifest_episodes.append(
                {
                    "new_episode_index": new_episode_index,
                    "recovery_type": episode.recovery_type,
                    "source_dataset_role": episode.dataset_role,
                    "source_dataset_dir": str(episode.dataset_dir),
                    "old_episode_index": episode.old_episode_index,
                    "source_episode": episode.source_episode,
                    "variant_index": episode.variant_index,
                    "raw_episode_name": episode.raw_episode_name,
                    "raw_episode_dir": str(episode.raw_episode_dir),
                    "frame_count": episode.frame_count,
                    "source_video_paths": source_videos,
                    "output_video_paths": output_videos,
                }
            )
            if (new_episode_index + 1) % 50 == 0 or new_episode_index + 1 == len(selected):
                logging.info(
                    "已处理episode %d/%d，累计frames=%d。",
                    new_episode_index + 1,
                    len(selected),
                    global_index,
                )

        merged_table = pa.concat_tables(episode_tables)
        if merged_table.num_rows != total_frames or global_index != total_frames:
            raise RuntimeError(
                f"合并帧数不一致: table={merged_table.num_rows}, "
                f"index={global_index}, expected={total_frames}"
            )
        parquet_path = staging_dir / "data" / "train-00000-of-00001.parquet"
        pq.write_table(merged_table, parquet_path, compression="snappy")

        save_file(
            {
                "from": torch.tensor(episode_from, dtype=torch.int64),
                "to": torch.tensor(episode_to, dtype=torch.int64),
            },
            staging_dir / "meta_data" / "episode_data_index.safetensors",
        )
        merged_stats = build_merged_stats(
            table=merged_table,
            camera_keys=list(sources[args.original_source].info["camera_keys"]),
            sources=sources,
        )
        save_file(merged_stats, staging_dir / "meta_data" / "stats.safetensors")
        write_support_files(
            staging_dir=staging_dir,
            sources=sources,
            original_source=args.original_source,
            total_episodes=len(selected),
            total_frames=total_frames,
            composition=composition,
        )
        write_json(
            staging_dir / "recovery_manifest.json",
            {
                "schema_version": 2 if "mixed" in sources else 1,
                "merge_type": (
                    "deduplicated_arm_view_mixed_recovery"
                    if "mixed" in sources
                    else "deduplicated_arm_view_recovery"
                ),
                "original_source": args.original_source,
                "source_datasets": {
                    role: str(source.dataset_dir)
                    for role, source in sources.items()
                },
                "total_episodes": len(selected),
                "total_frames": total_frames,
                "composition": composition,
                "episodes": manifest_episodes,
            },
        )

        # 最终落盘前重新读取关键文件，防止生成半损坏数据集。
        written_table = pq.read_table(parquet_path)
        if written_table.num_rows != total_frames:
            raise RuntimeError(
                f"Parquet回读帧数不一致: {written_table.num_rows} != {total_frames}"
            )
        expected_video_count = len(selected) * len(
            sources[args.original_source].info["camera_keys"]
        )
        actual_video_count = sum(1 for _ in (staging_dir / "videos").glob("*.mp4"))
        if actual_video_count != expected_video_count:
            raise RuntimeError(
                f"视频数量不一致: {actual_video_count} != {expected_video_count}"
            )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    logging.info("合并完成: %s", output_dir)
    logging.info(
        "训练配置可设置 dataset_local_dir=%s",
        output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir,
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "去重并合并同一任务的Arm/View恢复LeRobot数据集，"
            "并可选加入Mixed恢复数据集。"
        )
    )
    parser.add_argument(
        "--arm-dir",
        default="outputs/5_hf_datasets/InsertPeg-3Arms/expert_100/quest_teleop_InsertPeg-3Arms-v0_rgb_arm_random_static_3+0",
        help="Arm恢复HF数据集目录。",
    )
    parser.add_argument(
        "--view-dir",
        default="outputs/5_hf_datasets/InsertPeg-3Arms/expert_100/quest_teleop_InsertPeg-3Arms-v0_rgb_view_random_static_3+0",
        help="View恢复HF数据集目录。",
    )
    parser.add_argument(
        "--mixed-dir",
        default="",
        help=(
            "可选的Mixed恢复HF数据集目录；传入后合并结果为"
            "专家+Arm+View+Mixed，不传则保持原Arm+View合并行为。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/5_hf_datasets/InsertPeg-3Arms/expert_100/quest_teleop_InsertPeg-3Arms-v0_rgb_arm_view_random_static_3+0",
        help=(
            "合并后的单一HF数据集目录；不指定时根据是否传入Mixed自动使用"
            "arm_view或arm_view_mixed名称。"
        ),
    )
    parser.add_argument(
        "--original-source",
        choices=("arm", "view", "mixed"),
        default="view",
        help="重复原始专家轨迹保留哪个来源；默认保留View数据集中的副本。",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="输出视频的组织方式；hardlink失败时自动退回copy。",
    )
    parser.add_argument(
        "--verify-originals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="去重前是否逐条确认所有来源的原始轨迹状态、动作和时间戳完全一致。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查输入并打印合并规模，不写输出数据。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的output-dir。",
    )
    return parser


def main() -> None:
    init_logging()
    args = build_arg_parser().parse_args()
    merge_datasets(args)


if __name__ == "__main__":
    main()
