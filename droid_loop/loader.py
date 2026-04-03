"""
DROID dataset loader.

lerobot/droid_1.0.1 stores images as encoded .mp4 video files — one per camera
per episode. The standard HuggingFace `datasets` streaming only returns scalar
columns (joint states, actions, metadata). This module stitches them together:

  1. Stream scalar rows with `datasets` (fast, no download of videos)
  2. For each episode, download the corresponding video file(s) via
     huggingface_hub (cached on disk after first download)
  3. Decode video frames with PyAV (av) and merge into the frame dicts
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from pyarrow import parquet as pq

DROID_DATASET = "lerobot/droid_1.0.1"

# Fallback camera keys — overridden by meta/info.json when available
_DEFAULT_IMAGE_KEYS = [
    "observation.images.wrist_left",
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
]


@dataclass(frozen=True)
class VideoRef:
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


_episode_video_index_cache: dict[str, dict[int, dict[str, VideoRef]]] = {}


def dataset_fps(dataset_id: str = DROID_DATASET) -> float:
    """Return dataset FPS from metadata (falls back to 15)."""
    meta = _load_meta(dataset_id)
    try:
        return float(meta.get("fps", 15))
    except (TypeError, ValueError):
        return 15.0


def _load_meta(dataset_id: str) -> dict:
    """Download meta/info.json from the dataset repo and return it as a dict."""
    try:
        local = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename="meta/info.json",
        )
        meta = json.loads(Path(local).read_text())
        # Print key structural fields so we can verify path assumptions
        print(f"[loader] info.json keys: {list(meta.keys())}")
        print(f"[loader] episodes_per_chunk: {meta.get('episodes_per_chunk')}")
        print(f"[loader] total_episodes: {meta.get('total_episodes')}")
        vid_features = {k: v for k, v in meta.get("features", {}).items()
                        if isinstance(v, dict) and v.get("_type") == "VideoFrame"}
        print(f"[loader] VideoFrame features: {list(vid_features.keys())}")
        # Print path template if present inside any VideoFrame feature
        for k, v in vid_features.items():
            if "path" in v:
                print(f"[loader] path template ({k}): {v['path']}")
        return meta
    except Exception as e:
        print(f"[loader] could not load meta/info.json ({e}), using defaults")
        return {}


def _video_keys(meta: dict) -> list[str]:
    """Return image/video keys from metadata."""
    keys: list[str] = []
    features = meta.get("features", {})
    for key, val in features.items():
        if not isinstance(val, dict):
            continue
        if "video.path" in val or "video.fps" in val:
            keys.append(key)
    return keys


def _decode_video_window(
    local_path: str,
    start_seconds: float,
    end_seconds: float,
    fps_hint: float,
) -> list[Image.Image]:
    """Decode a time-window from a local video file into PIL Images."""
    frames: list[Image.Image] = []

    start_idx = max(0, int(round(start_seconds * fps_hint)))
    end_idx = max(start_idx + 1, int(round(end_seconds * fps_hint)))

    with av.open(local_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            frames.append(frame.to_image().convert("RGB"))
    return frames


def _video_key_from_meta_col(col_name: str) -> str | None:
    prefix = "videos/"
    suffixes = ("/chunk_index", "/file_index", "/from_timestamp", "/to_timestamp")
    if not col_name.startswith(prefix):
        return None
    for suffix in suffixes:
        if col_name.endswith(suffix):
            return col_name[len(prefix) : -len(suffix)]
    return None


def _load_episode_video_index(dataset_id: str) -> dict[int, dict[str, VideoRef]]:
    """Load episode -> video-key -> video-ref mappings from meta/episodes parquet files."""
    if dataset_id in _episode_video_index_cache:
        return _episode_video_index_cache[dataset_id]

    files = list_repo_files(dataset_id, repo_type="dataset")
    episode_meta_files = sorted(
        p for p in files if p.startswith("meta/episodes/") and p.endswith(".parquet")
    )
    if not episode_meta_files:
        raise RuntimeError("No meta/episodes parquet files found in dataset repo")

    index: dict[int, dict[str, VideoRef]] = {}
    required_cols = ["episode_index"]

    for remote_path in episode_meta_files:
        local_path = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=remote_path,
        )
        table = pq.read_table(local_path)
        video_keys = sorted(
            {k for n in table.column_names if (k := _video_key_from_meta_col(n)) is not None}
        )
        cols = required_cols + [
            f"videos/{key}/{suffix}"
            for key in video_keys
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        ]
        table = pq.read_table(local_path, columns=cols)
        data = table.to_pydict()

        for row_idx, ep in enumerate(data["episode_index"]):
            episode_id = int(ep)
            refs: dict[str, VideoRef] = {}
            for key in video_keys:
                refs[key] = VideoRef(
                    chunk_index=int(data[f"videos/{key}/chunk_index"][row_idx]),
                    file_index=int(data[f"videos/{key}/file_index"][row_idx]),
                    from_timestamp=float(data[f"videos/{key}/from_timestamp"][row_idx]),
                    to_timestamp=float(data[f"videos/{key}/to_timestamp"][row_idx]),
                )
            index[episode_id] = refs

    _episode_video_index_cache[dataset_id] = index
    print(f"[loader] episode video index loaded: {len(index)} episodes")
    return index


def _normalize_image_key(image_key: str) -> str:
    aliases = {
        "observation.images.wrist_image_left": "observation.images.wrist_left",
        "observation.images.exterior_image_1_left": "observation.images.exterior_1_left",
        "observation.images.exterior_image_2_left": "observation.images.exterior_2_left",
    }
    return aliases.get(image_key, image_key)


def _fetch_episode_videos(
    dataset_id: str,
    episode_id: int,
    image_keys: list[str],
    fps: float,
) -> dict[str, list[Image.Image]]:
    """Download and decode all camera videos for one episode using metadata refs."""
    decoded: dict[str, list[Image.Image]] = {}
    missing_keys: list[str] = []
    index = _load_episode_video_index(dataset_id)
    refs = index.get(episode_id)
    if refs is None:
        print(f"[loader] warning: no episode metadata for episode_id={episode_id}")
        return decoded

    for key in image_keys:
        ref = refs.get(key)
        if ref is None:
            missing_keys.append(key)
            continue

        hf_path = f"videos/{key}/chunk-{ref.chunk_index:03d}/file-{ref.file_index:03d}.mp4"
        try:
            local_path = hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename=hf_path,
            )
            decoded[key] = _decode_video_window(
                local_path,
                start_seconds=ref.from_timestamp,
                end_seconds=ref.to_timestamp,
                fps_hint=fps,
            )
            print(
                f"[loader] ep {episode_id} {key}: {len(decoded[key])} frames "
                f"(chunk-{ref.chunk_index:03d}/file-{ref.file_index:03d})"
            )
        except Exception as exc:
            missing_keys.append(key)
            print(
                f"[loader] warning: could not load {key} for ep {episode_id} "
                f"(chunk-{ref.chunk_index:03d}/file-{ref.file_index:03d}): {exc}"
            )

    if missing_keys:
        print(
            f"[loader] ep {episode_id}: missing {len(missing_keys)}/{len(image_keys)} camera files "
            f"({', '.join(missing_keys)})"
        )
    return decoded


def stream_episodes(
    dataset_id: str = DROID_DATASET,
    split: str = "train",
    image_keys: list[str] | None = None,
    start_episode: int = 0,
    max_episodes: int | None = None,
) -> Iterator[list[dict]]:
    """Stream DROID frames grouped by episode.

    Yields one list[dict] per episode. Each dict has all scalar fields from the
    parquet row plus camera image keys mapped to PIL Images.
    """
    meta = _load_meta(dataset_id)
    selected_keys = image_keys or (_video_keys(meta) or _DEFAULT_IMAGE_KEYS)
    image_keys = [_normalize_image_key(k) for k in selected_keys]
    fps = float(meta.get("fps", 15))

    print(f"[loader] {dataset_id}")
    print(f"[loader] image keys  : {image_keys}")
    print(f"[loader] fps         : {fps}")

    # Stream scalar rows grouped by episode
    ds = load_dataset(dataset_id, split=split, streaming=True)

    current_ep: list[dict] = []
    current_id: int | None = None
    yielded = 0

    def maybe_yield_episode(ep_rows: list[dict]) -> list[dict] | None:
        nonlocal yielded
        if not ep_rows:
            return None
        ep_id = int(ep_rows[0]["episode_index"])
        if ep_id < start_episode:
            return None
        if max_episodes is not None and yielded >= max_episodes:
            return None
        yielded += 1
        return _build_episode(ep_rows, dataset_id, image_keys, fps)

    for row in ds:
        ep_id = int(row["episode_index"])
        if current_id is None:
            current_id = ep_id
        if ep_id != current_id:
            episode = maybe_yield_episode(current_ep)
            if episode is not None:
                yield episode
            if max_episodes is not None and yielded >= max_episodes:
                return
            current_ep = []
            current_id = ep_id
        current_ep.append(row)

    if current_ep:
        episode = maybe_yield_episode(current_ep)
        if episode is not None:
            yield episode


def _build_episode(
    scalar_frames: list[dict],
    dataset_id: str,
    image_keys: list[str],
    fps: float,
) -> list[dict]:
    """Merge scalar metadata with decoded video frames for one episode."""
    ep_id = int(scalar_frames[0]["episode_index"])
    camera_frames = _fetch_episode_videos(dataset_id, ep_id, image_keys, fps)

    result: list[dict] = []
    for i, row in enumerate(scalar_frames):
        frame: dict = dict(row)
        # Ensure canonical integer fields
        frame["episode_index"] = ep_id
        frame["frame_index"] = int(row.get("frame_index", i))
        for key, frames in camera_frames.items():
            if i < len(frames):
                frame[key] = frames[i]
        result.append(frame)

    return result
