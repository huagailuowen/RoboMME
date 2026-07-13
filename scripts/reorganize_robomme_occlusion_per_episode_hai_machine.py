"""Reorganize RoboMME LeRobot v3 chunk files into v2.1 episode files.

This is a storage-only conversion. It never reruns simulation and uses H.264
stream copy, so video frames are not re-encoded. New data, videos, and metadata
are built in a staging directory and atomically installed only after validation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import imageio_ffmpeg
import pyarrow.parquet as pq


DEFAULT_DATA_ROOT = Path(
    "/home/yininghong/chenyuan/TTT-physics/repos/FastWAM-TTT/data/robomme-occlusion"
)
CASE_TYPES = ("sequence", "swap", "reveal")
VIDEO_KEYS = ("observation.images.image", "observation.images.wrist_image")
LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


def _dataset_root(data_root: Path, case_type: str) -> Path:
    return data_root / f"robomme_occlusion_{case_type}_missing05_hai-machine_lerobot"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")))
            stream.write("\n")


def _normalize_task_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        task = row.get("task", row.get("__index_level_0__"))
        if task is None:
            raise KeyError(f"Task text is missing from row: {row}")
        normalized.append({"task_index": int(row["task_index"]), "task": str(task)})
    normalized.sort(key=lambda row: row["task_index"])
    return normalized


def _repair_legacy_tasks(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    _write_jsonl(path, _normalize_task_rows(rows))


def _strip_new_huggingface_schema_metadata(data_root: Path) -> int:
    rewritten = 0
    for path in sorted(data_root.glob("chunk-*/episode_*.parquet")):
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.metadata:
            continue
        table = parquet.read().replace_schema_metadata(None)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
        rewritten += 1
    return rewritten


def _load_episode_rows(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No v3 episode index parquet found under {root}")
    rows = []
    for path in files:
        rows.extend(pq.ParquetFile(path).read().to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    expected = list(range(len(rows)))
    actual = [int(row["episode_index"]) for row in rows]
    if actual != expected:
        raise ValueError(f"Non-contiguous episode indices in {root}: {actual[:10]}")
    return rows


def _legacy_stats(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    allowed = {"min", "max", "mean", "std", "count"}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        feature_and_stat = key[len("stats/") :]
        feature, stat = feature_and_stat.rsplit("/", 1)
        if stat in allowed:
            result.setdefault(feature, {})[stat] = value
    return result


def _source_data_table(root: Path, episode_rows: list[dict[str, Any]]):
    locations = {
        (int(row["data/chunk_index"]), int(row["data/file_index"]))
        for row in episode_rows
    }
    if len(locations) != 1:
        raise ValueError(f"Expected one source data file in {root}, found {locations}")
    chunk_index, file_index = locations.pop()
    path = root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
    return pq.ParquetFile(path).read()


def _write_episode_parquets(
    root: Path,
    staging: Path,
    episode_rows: list[dict[str, Any]],
    chunks_size: int,
) -> None:
    source = _source_data_table(root, episode_rows)
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        expected_length = int(row["length"])
        if end - start != expected_length:
            raise ValueError(
                f"Episode {episode_index}: index span {end - start} != length {expected_length}"
            )
        table = source.slice(start, expected_length).replace_schema_metadata(None)
        destination = (
            staging
            / "data"
            / f"chunk-{episode_index // chunks_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")


def _source_video_path(
    root: Path, episode_rows: list[dict[str, Any]], video_key: str
) -> Path:
    locations = {
        (
            int(row[f"videos/{video_key}/chunk_index"]),
            int(row[f"videos/{video_key}/file_index"]),
        )
        for row in episode_rows
    }
    if len(locations) != 1:
        raise ValueError(
            f"Expected one source video for {video_key} in {root}, found {locations}"
        )
    chunk_index, file_index = locations.pop()
    return root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def _split_video_stream_copy(
    root: Path,
    staging: Path,
    episode_rows: list[dict[str, Any]],
    video_key: str,
    chunks_size: int,
    fps: int,
) -> None:
    source = _source_video_path(root, episode_rows, video_key)
    if not source.is_file():
        raise FileNotFoundError(source)
    if len(episode_rows) >= chunks_size:
        raise ValueError("This converter currently expects all episodes in one legacy chunk")

    output_dir = staging / "videos" / "chunk-000" / video_key
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_times = [
        float(row[f"videos/{video_key}/from_timestamp"])
        for row in episode_rows[1:]
    ]
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_times",
        ",".join(f"{timestamp:.9f}" for timestamp in segment_times),
        "-segment_time_delta",
        f"{0.5 / fps:.9f}",
        "-reset_timestamps",
        "1",
        "-segment_start_number",
        "0",
        str(output_dir / "episode_%06d.mp4"),
    ]
    subprocess.run(command, check=True)


def _video_frame_count(path: Path) -> int:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def _normalize_video_start_time(path: Path) -> bool:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        start_time = int(stream.start_time or 0)
        expected_frames = int(stream.frames)
    if start_time == 0:
        return False

    temporary = path.with_suffix(".timestamp-normalized.mp4")
    with av.open(str(path), mode="r") as input_container, av.open(
        str(temporary), mode="w", format="mp4"
    ) as output_container:
        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream_from_template(input_stream)
        for packet in input_container.demux(input_stream):
            if packet.pts is None or packet.dts is None:
                continue
            packet.pts -= start_time
            packet.dts -= start_time
            packet.stream = output_stream
            output_container.mux(packet)

    with av.open(str(temporary), mode="r") as container:
        stream = container.streams.video[0]
        if int(stream.start_time or 0) != 0:
            temporary.unlink()
            raise ValueError(f"Failed to normalize video start time for {path}")
        if expected_frames and int(stream.frames) != expected_frames:
            temporary.unlink()
            raise ValueError(
                f"Timestamp remux changed frame count for {path}: "
                f"{stream.frames} != {expected_frames}"
            )
    os.replace(temporary, path)
    return True


def _normalize_video_tree(video_root: Path) -> int:
    return sum(
        int(_normalize_video_start_time(path))
        for path in sorted(video_root.glob("chunk-*/*/episode_*.mp4"))
    )


def _validate_staging(
    staging: Path,
    episode_rows: list[dict[str, Any]],
    chunks_size: int,
) -> dict[str, int]:
    parquet_count = 0
    video_count = 0
    total_frames = 0
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        expected_length = int(row["length"])
        chunk = episode_index // chunks_size
        parquet_path = (
            staging / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.ParquetFile(parquet_path).read(
            columns=["episode_index", "frame_index"]
        )
        if table.num_rows != expected_length:
            raise ValueError(
                f"{parquet_path}: {table.num_rows} rows != {expected_length}"
            )
        episode_values = table.column("episode_index").to_pylist()
        frame_values = table.column("frame_index").to_pylist()
        if set(episode_values) != {episode_index}:
            raise ValueError(f"{parquet_path}: incorrect episode_index values")
        if frame_values != list(range(expected_length)):
            raise ValueError(f"{parquet_path}: non-contiguous frame_index values")
        parquet_count += 1
        total_frames += expected_length

        for video_key in VIDEO_KEYS:
            video_path = (
                staging
                / "videos"
                / f"chunk-{chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            _normalize_video_start_time(video_path)
            actual_frames = _video_frame_count(video_path)
            if actual_frames != expected_length:
                raise ValueError(
                    f"{video_path}: {actual_frames} frames != {expected_length}"
                )
            video_count += 1

    return {
        "episodes": len(episode_rows),
        "parquet_files": parquet_count,
        "video_files": video_count,
        "frames": total_frames,
    }


def _write_legacy_metadata(
    root: Path,
    staging: Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    meta_dir = staging / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = [
        key for key, feature in info["features"].items() if feature["dtype"] == "video"
    ]

    legacy_info = dict(info)
    legacy_info["codebase_version"] = "v2.1"
    legacy_info["total_videos"] = len(episode_rows) * len(video_keys)
    legacy_info["total_chunks"] = (len(episode_rows) + chunks_size - 1) // chunks_size
    legacy_info["data_path"] = LEGACY_DATA_PATH
    legacy_info["video_path"] = LEGACY_VIDEO_PATH
    legacy_info.pop("data_files_size_in_mb", None)
    legacy_info.pop("video_files_size_in_mb", None)
    for key in video_keys:
        feature = legacy_info["features"][key]
        shape = list(feature["shape"])
        names = list(feature.get("names") or [])
        if names[:3] == ["height", "width", "channel"]:
            feature["shape"] = [shape[2], shape[0], shape[1]]
            feature["names"] = ["channel", "height", "width"]
    _write_json(meta_dir / "info.json", legacy_info)

    episode_json = [
        {
            "episode_index": int(row["episode_index"]),
            "tasks": list(row["tasks"]),
            "length": int(row["length"]),
        }
        for row in episode_rows
    ]
    stats_json = [
        {
            "episode_index": int(row["episode_index"]),
            "stats": _legacy_stats(row),
        }
        for row in episode_rows
    ]
    tasks = _normalize_task_rows(
        pq.ParquetFile(root / "meta" / "tasks.parquet").read().to_pylist()
    )
    _write_jsonl(meta_dir / "episodes.jsonl", episode_json)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", stats_json)
    _write_jsonl(meta_dir / "tasks.jsonl", tasks)
    return legacy_info


def _install_staging(root: Path, staging: Path) -> None:
    names = ("data", "videos", "meta")
    backups = {name: root / f".{name}.v3_chunked_backup" for name in names}
    if any(path.exists() for path in backups.values()):
        raise FileExistsError(f"A previous backup still exists under {root}")

    moved_old = []
    installed_new = []
    try:
        for name in names:
            os.replace(root / name, backups[name])
            moved_old.append(name)
        for name in names:
            os.replace(staging / name, root / name)
            installed_new.append(name)
    except Exception:
        for name in reversed(installed_new):
            if (root / name).exists():
                shutil.rmtree(root / name)
        for name in reversed(moved_old):
            if backups[name].exists():
                os.replace(backups[name], root / name)
        raise

    for path in backups.values():
        shutil.rmtree(path)
    staging.rmdir()


def reorganize_dataset(root: Path) -> dict[str, Any]:
    generation_metadata_path = root / "robomme_occlusion_generation_metadata.json"
    generation_metadata = _read_json(generation_metadata_path)
    if generation_metadata.get("generation_status") != "completed":
        raise RuntimeError(
            f"Refusing to reorganize incomplete dataset {root}: "
            f"status={generation_metadata.get('generation_status')}"
        )

    info = _read_json(root / "meta" / "info.json")
    if info.get("codebase_version") == "v2.1" and "episode_{episode_index" in info.get(
        "video_path", ""
    ):
        _repair_legacy_tasks(root / "meta" / "tasks.jsonl")
        rewritten = _strip_new_huggingface_schema_metadata(root / "data")
        normalized_videos = _normalize_video_tree(root / "videos")
        print(
            f"[{root.name}] already uses per-episode v2.1 layout; "
            f"normalized_schema_files={rewritten}; "
            f"normalized_video_timestamps={normalized_videos}",
            flush=True,
        )
        return {
            "episodes": int(info["total_episodes"]),
            "parquet_files": int(info["total_episodes"]),
            "video_files": int(info["total_videos"]),
            "frames": int(info["total_frames"]),
        }

    episode_rows = _load_episode_rows(root)
    if len(episode_rows) != int(info["total_episodes"]):
        raise ValueError(
            f"{root}: metadata has {len(episode_rows)} episodes, info has {info['total_episodes']}"
        )
    chunks_size = int(info.get("chunks_size", 1000))
    staging = root / ".per_episode_reorganize_tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        print(f"[{root.name}] splitting {len(episode_rows)} parquet episodes", flush=True)
        _write_episode_parquets(root, staging, episode_rows, chunks_size)
        for video_key in VIDEO_KEYS:
            print(f"[{root.name}] stream-copy splitting {video_key}", flush=True)
            _split_video_stream_copy(
                root, staging, episode_rows, video_key, chunks_size, int(info["fps"])
            )
        legacy_info = _write_legacy_metadata(root, staging, info, episode_rows)
        summary = _validate_staging(staging, episode_rows, chunks_size)
        if summary["frames"] != int(legacy_info["total_frames"]):
            raise ValueError(
                f"Validated {summary['frames']} frames, expected {legacy_info['total_frames']}"
            )
        _install_staging(root, staging)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    generation_metadata["lerobot_storage_layout"] = "v2.1_per_episode"
    generation_metadata["reorganized_from"] = "v3.0_chunked_files"
    generation_metadata["reorganized_at"] = datetime.now().isoformat()
    generation_metadata["per_episode_layout_validation"] = summary
    _write_json(generation_metadata_path, generation_metadata)
    print(f"[{root.name}] completed: {summary}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--case-types", nargs="+", choices=CASE_TYPES, default=list(CASE_TYPES)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    for case_type in args.case_types:
        root = _dataset_root(args.data_root, case_type)
        results[case_type] = reorganize_dataset(root)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
