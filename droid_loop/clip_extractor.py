from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image


DEFAULT_TIMESTAMP_KEYS = (
    "timestamp",
    "timestamp_s",
    "time",
    "observation.timestamp",
    "observation.time",
)


@dataclass(frozen=True)
class ClipRequest:
    """Request to extract a temporal clip around an anchor event."""

    anchor_timestamp_s: float | None = None
    anchor_frame_index: int | None = None
    pre_s: float = 2.0
    post_s: float = 2.0
    max_frames: int = 16
    strategy: Literal["uniform", "dense_center"] = "uniform"


@dataclass(frozen=True)
class ClipFrame:
    frame_index: int
    timestamp_s: float
    is_anchor: bool


@dataclass(frozen=True)
class ExtractedClip:
    anchor_frame_index: int
    anchor_timestamp_s: float
    start_frame_index: int
    end_frame_index: int
    start_timestamp_s: float
    end_timestamp_s: float
    frames: list[ClipFrame]


def _as_pil_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    return Image.fromarray(value)


def _timestamp_for_row(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def frame_timestamps(
    episode_rows: list[dict],
    fps: float,
    timestamp_keys: tuple[str, ...] = DEFAULT_TIMESTAMP_KEYS,
) -> list[float]:
    """Return per-row timestamps in seconds.

    Falls back to frame_index/fps (or row offset/fps) when explicit timestamps
    are unavailable.
    """
    safe_fps = max(0.001, float(fps))
    out: list[float] = []
    for i, row in enumerate(episode_rows):
        ts = _timestamp_for_row(row, timestamp_keys)
        if ts is not None:
            out.append(ts)
            continue
        fi = int(row.get("frame_index", i))
        out.append(fi / safe_fps)
    return out


def _resolve_anchor_position(
    episode_rows: list[dict],
    timestamps_s: list[float],
    request: ClipRequest,
) -> int:
    if request.anchor_frame_index is not None:
        target_fi = int(request.anchor_frame_index)
        for pos, row in enumerate(episode_rows):
            if int(row.get("frame_index", pos)) == target_fi:
                return pos
        # Fallback to nearest frame index if exact one is absent.
        nearest = min(
            range(len(episode_rows)),
            key=lambda pos: abs(int(episode_rows[pos].get("frame_index", pos)) - target_fi),
        )
        return int(nearest)

    if request.anchor_timestamp_s is not None:
        target_ts = float(request.anchor_timestamp_s)
        nearest = min(
            range(len(timestamps_s)),
            key=lambda pos: abs(timestamps_s[pos] - target_ts),
        )
        return int(nearest)

    return len(episode_rows) // 2


def _uniform_sample_positions(positions: list[int], n: int) -> list[int]:
    if n >= len(positions):
        return positions
    idx = np.linspace(0, len(positions) - 1, num=n, dtype=int).tolist()
    return [positions[i] for i in idx]


def _dense_center_sample_positions(
    positions: list[int], n: int, anchor_pos: int
) -> list[int]:
    if n >= len(positions):
        return positions
    sorted_by_center = sorted(positions, key=lambda p: abs(p - anchor_pos))
    chosen = sorted(sorted_by_center[:n])
    return chosen


def _sample_positions(
    positions: list[int], n: int, strategy: str, anchor_pos: int
) -> list[int]:
    if n <= 0:
        return []
    if strategy == "dense_center":
        return _dense_center_sample_positions(positions, n, anchor_pos)
    return _uniform_sample_positions(positions, n)


def extract_clip(
    episode_rows: list[dict],
    request: ClipRequest,
    fps: float,
    timestamp_keys: tuple[str, ...] = DEFAULT_TIMESTAMP_KEYS,
) -> tuple[ExtractedClip, list[dict]]:
    """Extract a temporal clip around an anchor event.

    Returns:
      - clip metadata + selected frame descriptors
      - selected source rows (same order as descriptors)
    """
    if not episode_rows:
        raise ValueError("episode_rows cannot be empty")

    timestamps_s = frame_timestamps(episode_rows, fps=fps, timestamp_keys=timestamp_keys)
    anchor_pos = _resolve_anchor_position(episode_rows, timestamps_s, request)
    anchor_ts = timestamps_s[anchor_pos]
    start_ts = max(timestamps_s[0], anchor_ts - max(0.0, request.pre_s))
    end_ts = min(timestamps_s[-1], anchor_ts + max(0.0, request.post_s))

    window_positions = [
        pos for pos, ts in enumerate(timestamps_s) if start_ts <= ts <= end_ts
    ]
    if not window_positions:
        window_positions = [anchor_pos]
    sampled_positions = _sample_positions(
        positions=window_positions,
        n=max(1, int(request.max_frames)),
        strategy=request.strategy,
        anchor_pos=anchor_pos,
    )
    if anchor_pos not in sampled_positions:
        sampled_positions.append(anchor_pos)
        sampled_positions = sorted(set(sampled_positions))
    if len(sampled_positions) > request.max_frames:
        sampled_positions = _sample_positions(
            positions=sampled_positions,
            n=max(1, int(request.max_frames)),
            strategy=request.strategy,
            anchor_pos=anchor_pos,
        )
        if anchor_pos not in sampled_positions:
            sampled_positions[-1] = anchor_pos
            sampled_positions = sorted(set(sampled_positions))

    selected_rows = [episode_rows[pos] for pos in sampled_positions]
    frames = [
        ClipFrame(
            frame_index=int(episode_rows[pos].get("frame_index", pos)),
            timestamp_s=float(timestamps_s[pos]),
            is_anchor=(pos == anchor_pos),
        )
        for pos in sampled_positions
    ]

    clip = ExtractedClip(
        anchor_frame_index=int(episode_rows[anchor_pos].get("frame_index", anchor_pos)),
        anchor_timestamp_s=float(anchor_ts),
        start_frame_index=int(episode_rows[sampled_positions[0]].get("frame_index", sampled_positions[0])),
        end_frame_index=int(episode_rows[sampled_positions[-1]].get("frame_index", sampled_positions[-1])),
        start_timestamp_s=float(timestamps_s[sampled_positions[0]]),
        end_timestamp_s=float(timestamps_s[sampled_positions[-1]]),
        frames=frames,
    )
    return clip, selected_rows


def extract_clip_images(
    episode_rows: list[dict],
    image_key: str,
    request: ClipRequest,
    fps: float,
    timestamp_keys: tuple[str, ...] = DEFAULT_TIMESTAMP_KEYS,
) -> tuple[ExtractedClip, list[Image.Image]]:
    """Extract clip metadata and sampled PIL images for a single camera key."""
    clip, rows = extract_clip(
        episode_rows=episode_rows,
        request=request,
        fps=fps,
        timestamp_keys=timestamp_keys,
    )
    images: list[Image.Image] = []
    for row in rows:
        value = row.get(image_key)
        if value is None:
            continue
        images.append(_as_pil_image(value).convert("RGB"))
    return clip, images
