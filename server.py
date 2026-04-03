import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

load_dotenv()

from droid_loop import catalog, labeler, loader, signal
from droid_loop.clip_extractor import ClipRequest, extract_clip, frame_timestamps
from droid_loop.scorer import FrameScorer
from droid_loop.vlm_query import MODEL_ID as VLM_MODEL_ID, query_incident

SERVER_ROOT = Path(__file__).resolve().parent

# Absolute runtime paths anchored to the repo so they do not depend on cwd.
FRAMES_DIR = SERVER_ROOT / "frames"
FRAMES_DIR.mkdir(exist_ok=True)
print(f"[droid-loop] frames dir: {FRAMES_DIR}")

app = FastAPI(title="DROID Loop API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CATALOG_PATH = catalog.DB_PATH

# Touch the DB on startup so first-run queries don't fail.
DESCRIPTIONS_PATH = SERVER_ROOT / "cluster_descriptions.json"
SIGNAL_PATH = SERVER_ROOT / "training_signal.json"


def _migrate_runtime_file(legacy_name: str, target: Path) -> None:
    legacy = Path.cwd() / legacy_name
    if legacy.resolve() == target.resolve():
        return
    if target.exists() or not legacy.exists():
        return
    legacy.replace(target)


def _migrate_runtime_files() -> None:
    _migrate_runtime_file("catalog.db", CATALOG_PATH)
    _migrate_runtime_file("cluster_descriptions.json", DESCRIPTIONS_PATH)
    _migrate_runtime_file("training_signal.json", SIGNAL_PATH)


_migrate_runtime_files()
catalog._db(CATALOG_PATH)


# ── Job state ─────────────────────────────────────────────────────────────────
# One job at a time. Progress events are pushed to a shared queue and
# streamed to the UI via SSE.

_job_lock = threading.Lock()
_job_running: dict = {"name": None}
_event_subscribers_lock = threading.Lock()
_event_subscribers: set[queue.Queue] = set()
_process = psutil.Process()
_episode_scores: dict[int, list[dict]] = {}
_episode_videos: dict[int, str] = {}
_episode_camera_videos: dict[int, dict[str, str]] = {}
_server_stop_event = threading.Event()
_scan_pause_event = threading.Event()
_scan_cancel_event = threading.Event()
_label_cancel_event = threading.Event()


def _probe_video_frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return 0
    if result.returncode != 0:
        return 0
    try:
        return max(0, int((result.stdout or "").strip() or "0"))
    except ValueError:
        return 0


def _synthetic_episode_score_trace(episode_id: int) -> list[dict]:
    catalog.set_episode_score_state(episode_id, "scoring", path=CATALOG_PATH)
    media = catalog.get_episode_media(episode_id, CATALOG_PATH)
    candidate_urls: list[str] = []
    stored_video = media.get("video_url")
    if isinstance(stored_video, str) and stored_video:
        candidate_urls.append(stored_video)
    stored_cameras = media.get("camera_videos")
    if isinstance(stored_cameras, dict):
        candidate_urls.extend(str(url) for url in stored_cameras.values() if isinstance(url, str))
    candidate_paths = [FRAMES_DIR / Path(url).name for url in candidate_urls]
    if not candidate_paths:
        candidate_paths = sorted(FRAMES_DIR.glob(f"episode_{episode_id}__*.mp4"))
    candidate_paths = [path for path in candidate_paths if _video_file_is_usable(path)]
    if not candidate_paths:
        catalog.set_episode_score_state(episode_id, "failed", "no episode media available", CATALOG_PATH)
        return []
    frame_count = max(_probe_video_frame_count(path) for path in candidate_paths)
    if frame_count <= 1:
        catalog.set_episode_score_state(episode_id, "failed", "episode media has <=1 frame", CATALOG_PATH)
        return []

    moments = catalog.get_episode_moments(episode_id, CATALOG_PATH)
    moment_by_frame = {int(m["frame_index"]): m for m in moments}
    flagged_frames = sorted(moment_by_frame)
    highlight_frames = set(flagged_frames)
    for frame_index in flagged_frames:
        for offset in (-2, -1, 1, 2):
            neighbor = frame_index + offset
            if 0 <= neighbor < frame_count:
                highlight_frames.add(neighbor)

    trace: list[dict] = []
    for frame_index in range(frame_count):
        moment = moment_by_frame.get(frame_index)
        flagged = frame_index in moment_by_frame
        in_window = frame_index in highlight_frames
        trace.append(
            {
                "frame_index": frame_index,
                "score": 0.58 if flagged else (0.22 if in_window else 0.08),
                "cluster_id": int(moment["cluster_id"]) if moment else -1,
                "flagged": flagged,
                "camera_scores": {},
                "camera_flags": {},
                "camera_clusters": {},
            }
        )

    catalog.save_episode_scores(episode_id=episode_id, score_trace=trace, path=CATALOG_PATH)
    if episode_id in catalog.get_scanned_episode_ids(CATALOG_PATH):
        catalog.mark_episode_scanned(
            episode_id=episode_id,
            frame_count=frame_count,
            flagged_count=len(flagged_frames),
            path=CATALOG_PATH,
        )
    _episode_scores[episode_id] = trace
    return trace
_scan_thread: threading.Thread | None = None
_label_thread: threading.Thread | None = None
_vlm_query_cache: dict[str, dict] = {}
_vlm_token_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
_scan_telemetry_lock = threading.Lock()
_scan_telemetry: dict[str, object] = {
    "active": False,
    "paused": False,
    "started_at": None,
    "queue_size": 0,
    "queue_capacity": 0,
    "current_episode": None,
    "loaded_episodes": 0,
    "scored_episodes": 0,
    "loaded_frames": 0,
    "scored_frames": 0,
    "total_flagged": 0,
}


def _push(event: dict) -> None:
    critical = event.get("type") in {"progress", "done", "error"}
    with _event_subscribers_lock:
        subscribers = list(_event_subscribers)
    for subscriber in subscribers:
        try:
            subscriber.put_nowait(event)
        except queue.Full:
            if not critical:
                continue
            try:
                subscriber.get_nowait()
            except queue.Empty:
                pass
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                continue


def _clear_event_queue() -> None:
    with _event_subscribers_lock:
        subscribers = list(_event_subscribers)
    for subscriber in subscribers:
        while not subscriber.empty():
            try:
                subscriber.get_nowait()
            except queue.Empty:
                break


def _cleanup_runtime_state() -> None:
    _episode_scores.clear()
    _episode_videos.clear()
    _episode_camera_videos.clear()
    _set_scan_telemetry(active=False, paused=False, current_episode=None, queue_size=0)
    with _job_lock:
        _job_running["name"] = None
    _clear_event_queue()


def _cleanup_artifacts() -> None:
    # Keep persisted review media so UI state can recover across server restarts.
    # Only purge transient scan preview artifacts on shutdown.
    for path in FRAMES_DIR.rglob("*"):
        if path.is_file():
            if not path.name.startswith("prev_"):
                continue
            try:
                path.unlink()
            except OSError:
                pass
    # Purge generated root artifacts as part of graceful shutdown cleanup.
    # Resolve both CWD-relative and server-root paths for determinism.
    server_root = Path(__file__).parent
    artifact_candidates = [
        DESCRIPTIONS_PATH,
        SIGNAL_PATH,
        server_root / DESCRIPTIONS_PATH.name,
        server_root / SIGNAL_PATH.name,
    ]
    for path in artifact_candidates:
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            pass


def _wait_for_active_job_stop(timeout_s: float = 1.0) -> None:
    deadline = time.time() + max(0.1, timeout_s)
    while time.time() < deadline:
        with _job_lock:
            if _job_running["name"] is None:
                return
        time.sleep(0.05)


def _join_worker_threads(timeout_s: float = 1.5) -> None:
    deadline = time.time() + max(0.1, timeout_s)
    global _scan_thread, _label_thread
    for thread_ref_name in ("_scan_thread", "_label_thread"):
        thread = _scan_thread if thread_ref_name == "_scan_thread" else _label_thread
        if thread is None:
            continue
        remaining = max(0.05, deadline - time.time())
        thread.join(timeout=remaining)
        if not thread.is_alive():
            if thread_ref_name == "_scan_thread":
                _scan_thread = None
            else:
                _label_thread = None


def _log_scan_event(event: str, **fields) -> None:
    payload = {"component": "scan", "event": event, **fields}
    print(json.dumps(payload, default=str))


def _rss_mb() -> float:
    return round(_process.memory_info().rss / (1024 * 1024), 2)


def _set_scan_telemetry(**fields: object) -> None:
    with _scan_telemetry_lock:
        _scan_telemetry.update(fields)


def _scan_status_snapshot() -> dict[str, object]:
    with _scan_telemetry_lock:
        snap = dict(_scan_telemetry)
    started_at = snap.get("started_at")
    if isinstance(started_at, float) and started_at > 0:
        elapsed_s = max(0.001, time.time() - started_at)
        loaded_episodes = int(snap.get("loaded_episodes", 0))
        scored_episodes = int(snap.get("scored_episodes", 0))
        loaded_frames = int(snap.get("loaded_frames", 0))
        scored_frames = int(snap.get("scored_frames", 0))
        snap["elapsed_s"] = round(elapsed_s, 2)
        snap["load_eps_per_min"] = round((loaded_episodes / elapsed_s) * 60.0, 2)
        snap["score_eps_per_min"] = round((scored_episodes / elapsed_s) * 60.0, 2)
        snap["load_frames_per_s"] = round(loaded_frames / elapsed_s, 2)
        snap["score_frames_per_s"] = round(scored_frames / elapsed_s, 2)
    else:
        snap["elapsed_s"] = 0.0
        snap["load_eps_per_min"] = 0.0
        snap["score_eps_per_min"] = 0.0
        snap["load_frames_per_s"] = 0.0
        snap["score_frames_per_s"] = 0.0
    snap["rss_mb"] = _rss_mb()
    return snap


def _wait_if_scan_paused() -> None:
    while not _scan_pause_event.is_set():
        if _server_stop_event.is_set() or _scan_cancel_event.is_set():
            break
        _set_scan_telemetry(paused=True)
        time.sleep(0.1)
    _set_scan_telemetry(paused=False)


def _scan_should_stop(local_stop: threading.Event | None = None) -> bool:
    if _server_stop_event.is_set() or _scan_cancel_event.is_set():
        return True
    return bool(local_stop is not None and local_stop.is_set())


def _merge_flag_positions(
    flagged_positions: list[int],
    frame_indices: list[int],
    scores: list[float],
    merge_gap_frames: int,
) -> tuple[list[int], list[dict[str, int]]]:
    if not flagged_positions:
        return [], []
    ordered = sorted({int(i) for i in flagged_positions})
    windows: list[dict[str, int]] = []
    run = [ordered[0]]
    gap = max(0, int(merge_gap_frames))
    for pos in ordered[1:]:
        prev = run[-1]
        frame_gap = int(frame_indices[pos]) - int(frame_indices[prev])
        if frame_gap <= gap:
            run.append(pos)
            continue
        best = max(run, key=lambda p: float(scores[p]))
        windows.append(
            {
                "anchor_pos": int(best),
                "start_frame": int(frame_indices[run[0]]),
                "end_frame": int(frame_indices[run[-1]]),
            }
        )
        run = [pos]
    best = max(run, key=lambda p: float(scores[p]))
    windows.append(
        {
            "anchor_pos": int(best),
            "start_frame": int(frame_indices[run[0]]),
            "end_frame": int(frame_indices[run[-1]]),
        }
    )
    return [int(w["anchor_pos"]) for w in windows], windows


def _sse_stream():
    subscriber: queue.Queue = queue.Queue(maxsize=2000)
    with _event_subscribers_lock:
        _event_subscribers.add(subscriber)
    try:
        while True:
            if _server_stop_event.is_set():
                break
            try:
                event = subscriber.get(timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                if _server_stop_event.is_set():
                    break
                with _job_lock:
                    if _job_running["name"] is None:
                        break
                yield 'data: {"type":"ping"}\n\n'
    finally:
        with _event_subscribers_lock:
            _event_subscribers.discard(subscriber)


# ── Frames ───────────────────────────────────────────────────────────────────

@app.get("/api/frames/{filename}")
def get_frame(filename: str):
    path = FRAMES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "Connection": "close",
        },
    )


@app.get("/api/videos/{filename}")
def get_video(filename: str):
    path = FRAMES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-store",
            "Connection": "close",
        },
    )


# ── Health / status ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    with _job_lock:
        running = _job_running["name"]
    return {"running": running, "scan": _scan_status_snapshot()}


# ── Catalog ───────────────────────────────────────────────────────────────────

@app.get("/api/catalog")
def get_catalog():
    if not CATALOG_PATH.exists():
        return []
    return list(catalog.load_moments(CATALOG_PATH))


# ── Episodes ──────────────────────────────────────────────────────────────────

@app.get("/api/episodes")
def get_episodes():
    if not CATALOG_PATH.exists():
        return []
    scanned = {
        row["episode_id"]: row
        for row in catalog.get_scanned_episodes(CATALOG_PATH)
    }
    moments = list(catalog.load_moments(CATALOG_PATH))
    for m in moments:
        ep = int(m["episode_id"])
        if ep not in scanned:
            scanned[ep] = {
                "episode_id": ep,
                "frame_count": 0,
                "flagged_count": 0,
                "scanned_at": 0.0,
            }
        scanned[ep]["flagged_count"] = int(scanned[ep].get("flagged_count", 0)) + 1
    return sorted(scanned.values(), key=lambda x: x["episode_id"])


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: int):
    if not CATALOG_PATH.exists():
        raise HTTPException(status_code=404, detail="Catalog not found")
    moments = catalog.get_episode_moments(episode_id, CATALOG_PATH)
    if not moments:
        raise HTTPException(status_code=404, detail="Episode not found")
    return moments


@app.get("/api/episode-scores/{episode_id}")
def get_episode_scores(episode_id: int):
    cached = _episode_scores.get(episode_id)
    if cached is not None and len(cached) >= 2:
        return cached
    stored = catalog.get_episode_scores(episode_id, CATALOG_PATH)
    if len(stored) >= 2:
        _episode_scores[episode_id] = stored
        return stored
    synthetic = _synthetic_episode_score_trace(episode_id)
    if len(synthetic) >= 2:
        return synthetic
    try:
        return _compute_episode_score_trace(episode_id)
    except HTTPException:
        return stored
    except Exception:
        return stored


@app.get("/api/episode-scores/{episode_id}/detail")
def get_episode_scores_detail(episode_id: int):
    record = catalog.get_episode_score_record(episode_id, CATALOG_PATH)
    trace = list(record.get("trace", []))
    if len(trace) >= 2:
        _episode_scores[episode_id] = trace
        return record
    synthetic = _synthetic_episode_score_trace(episode_id)
    if len(synthetic) >= 2:
        return catalog.get_episode_score_record(episode_id, CATALOG_PATH)
    try:
        rebuilt = _compute_episode_score_trace(episode_id)
        return {
            "state": "ready",
            "trace": rebuilt,
            "error": None,
            "updated_at": time.time(),
        }
    except HTTPException as exc:
        catalog.set_episode_score_state(episode_id, "failed", str(exc.detail), CATALOG_PATH)
        return catalog.get_episode_score_record(episode_id, CATALOG_PATH)
    except Exception as exc:
        catalog.set_episode_score_state(episode_id, "failed", str(exc), CATALOG_PATH)
        return catalog.get_episode_score_record(episode_id, CATALOG_PATH)


@app.get("/api/episode-video/{episode_id}")
def get_episode_video(episode_id: int):
    video_url = _episode_videos.get(episode_id)
    if not video_url:
        media = catalog.get_episode_media(episode_id, CATALOG_PATH)
        video_url = media.get("video_url")
        if isinstance(video_url, str) and video_url:
            _episode_videos[episode_id] = video_url
            cams = media.get("camera_videos")
            if isinstance(cams, dict):
                _episode_camera_videos[episode_id] = {
                    str(k): str(v) for k, v in cams.items()
                }
    if not video_url:
        cams = _episode_camera_videos.get(episode_id, {})
        if cams:
            video_url = next(iter(cams.values()))
    if video_url:
        path = FRAMES_DIR / Path(video_url).name
        if not _video_file_is_usable(path):
            video_url = None
    if not video_url:
        inferred = FRAMES_DIR / f"episode_{episode_id}.mp4"
        if _video_file_is_usable(inferred):
            video_url = f"/api/videos/{inferred.name}"
    if not video_url:
        inferred_cams = sorted(FRAMES_DIR.glob(f"episode_{episode_id}__*.mp4"))
        inferred_cams = [path for path in inferred_cams if _video_file_is_usable(path)]
        if inferred_cams:
            video_url = f"/api/videos/{inferred_cams[0].name}"
    if not video_url:
        video_url, _ = _ensure_episode_media_files(episode_id)
    return {"video_url": video_url}


@app.get("/api/episode-videos/{episode_id}")
def get_episode_videos(episode_id: int):
    cams = dict(_episode_camera_videos.get(episode_id, {}))
    if not cams:
        media = catalog.get_episode_media(episode_id, CATALOG_PATH)
        stored = media.get("camera_videos")
        if isinstance(stored, dict) and stored:
            cams = {str(k): str(v) for k, v in stored.items()}
            _episode_camera_videos[episode_id] = cams
        video_url = media.get("video_url")
        if isinstance(video_url, str) and video_url:
            _episode_videos[episode_id] = video_url
    if cams:
        cams = {
            camera: url
            for camera, url in cams.items()
            if _video_file_is_usable(FRAMES_DIR / Path(url).name)
        }
        if cams:
            _episode_camera_videos[episode_id] = cams
            return cams
    inferred: dict[str, str] = {}
    for path in sorted(FRAMES_DIR.glob(f"episode_{episode_id}__*.mp4")):
        if not _video_file_is_usable(path):
            continue
        camera = path.stem.split("__", 1)[-1]
        inferred[camera] = f"/api/videos/{path.name}"
    if inferred:
        return inferred
    primary = FRAMES_DIR / f"episode_{episode_id}.mp4"
    if _video_file_is_usable(primary):
        return {"primary": f"/api/videos/{primary.name}"}
    _, regenerated = _ensure_episode_media_files(episode_id)
    if regenerated:
        return regenerated
    primary = FRAMES_DIR / f"episode_{episode_id}.mp4"
    if _video_file_is_usable(primary):
        return {"primary": f"/api/videos/{primary.name}"}
    return {}


# ── Cluster descriptions ──────────────────────────────────────────────────────

@app.get("/api/descriptions")
def get_descriptions():
    if not DESCRIPTIONS_PATH.exists():
        return {}
    return json.loads(DESCRIPTIONS_PATH.read_text())


@app.get("/api/descriptions/{episode_id}")
def get_episode_descriptions(episode_id: int):
    if not DESCRIPTIONS_PATH.exists():
        return {}
    data = json.loads(DESCRIPTIONS_PATH.read_text())
    return data.get(str(episode_id), {})


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationPayload(BaseModel):
    episode_id: int
    frame_index: int
    cluster_id: int
    verdict: str  # "approved" | "rejected"


@app.post("/api/validate")
def validate_moment(payload: ValidationPayload):
    catalog.save_moment(
        {
            "episode_id": payload.episode_id,
            "frame_index": payload.frame_index,
            "cluster_id": payload.cluster_id,
            "label": payload.verdict,
        },
        CATALOG_PATH,
    )
    return {"status": "ok", "logged": payload.model_dump()}


# ── Scan ──────────────────────────────────────────────────────────────────────

class ScanParams(BaseModel):
    dataset_id: str = loader.DROID_DATASET
    contamination: float = 0.05
    image_key: str = "observation.images.wrist_left"
    context_window: int = 4
    start_episode: int = 0
    max_episodes: int | None = None
    max_frames_per_episode: int = 0  # 0 = no cap
    embed_batch_size: int = 32
    max_rss_mb: int = 12000
    incident_merge_gap_frames: int = 8


IMAGE_KEY_FALLBACKS = [
    "observation.images.wrist_left",
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    # Legacy aliases kept for older exports/checkpoints.
    "observation.images.wrist_image_left",
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_right",
]

CAMERA_TAGS = {
    "observation.images.wrist_left": "wrist_left",
    "observation.images.exterior_1_left": "exterior_1_left",
    "observation.images.exterior_2_left": "exterior_2_left",
    "observation.images.wrist_image_left": "wrist_left",
    "observation.images.exterior_image_1_left": "exterior_1_left",
    "observation.images.exterior_image_2_left": "exterior_2_left",
    "observation.images.wrist_image_right": "wrist_right",
}
REVERSE_CAMERA_TAGS = {v: k for k, v in CAMERA_TAGS.items()}
ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _resolve_image_key(frame: dict, preferred: str) -> str | None:
    """Return the first available image key in the frame, preferring the requested one."""
    if preferred in frame and frame[preferred] is not None:
        return preferred
    for key in IMAGE_KEY_FALLBACKS:
        if key in frame and frame[key] is not None:
            return key
    # Last resort: any key whose value is a numpy array or PIL Image
    detected = _find_image_keys(frame)
    if detected:
        print(f"[scan] using auto-detected image key: {detected[0]!r}  (all: {detected})")
        return detected[0]
    return None


def _find_image_keys(frame: dict) -> list[str]:
    """Return all keys in the frame that look like image data (numpy arrays or PIL Images)."""
    import numpy as np
    from PIL import Image as PILImage
    found = []
    for k, v in frame.items():
        if isinstance(v, (np.ndarray, PILImage.Image)):
            found.append(k)
        elif isinstance(v, dict):
            # nested dict — check one level deep (e.g. frame["observation"]["images"]["wrist"])
            for subk, subv in v.items():
                if isinstance(subv, (np.ndarray, PILImage.Image)):
                    found.append(f"{k}.{subk}")
    return found


def _as_pil_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    return Image.fromarray(value)


def _camera_tag(key: str) -> str:
    return CAMERA_TAGS.get(key, key.replace(".", "_"))


def _camera_key_from_tag_or_key(value: str) -> str:
    if value.startswith("observation.images."):
        return value
    return REVERSE_CAMERA_TAGS.get(value, value)


def _frame_camera_keys(frame: dict) -> list[str]:
    return [
        key for key, value in frame.items()
        if key.startswith("observation.images.") and value is not None
    ]


def _resolve_preview_key(frame: dict, preferred: str) -> str | None:
    if frame.get(preferred) is not None:
        return preferred
    detected = _frame_camera_keys(frame)
    return detected[0] if detected else None


def _frame_views_payload(episode_id: int, frame_index: int) -> list[dict[str, str]]:
    views: list[dict[str, str]] = []
    primary = FRAMES_DIR / f"{episode_id}_{frame_index}.jpg"
    if primary.exists():
        views.append({"camera": "primary", "frame_url": f"/api/frames/{primary.name}"})

    for path in sorted(FRAMES_DIR.glob(f"{episode_id}_{frame_index}__*.jpg")):
        name = path.stem
        if "__" not in name:
            continue
        _, cam = name.split("__", 1)
        views.append({"camera": cam, "frame_url": f"/api/frames/{path.name}"})
    return views


def _load_episode_for_query(dataset_id: str, episode_id: int) -> list[dict]:
    for episode in loader.stream_episodes(
        dataset_id=dataset_id,
        image_keys=[
            "observation.images.wrist_left",
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
        ],
        start_episode=episode_id,
        max_episodes=1,
    ):
        if int(episode[0]["episode_index"]) == int(episode_id):
            return episode
    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")


def _materialize_frame_files(
    episode_rows: list[dict],
    episode_id: int,
    frame_indices: list[int],
    preferred_image_key: str = "observation.images.wrist_left",
) -> None:
    wanted = set(int(fi) for fi in frame_indices)
    for row in episode_rows:
        fi = int(row.get("frame_index", -1))
        if fi not in wanted:
            continue
        primary = FRAMES_DIR / f"{episode_id}_{fi}.jpg"
        if not primary.exists():
            key = _resolve_preview_key(row, preferred_image_key)
            if key is not None:
                _as_pil_image(row[key]).convert("RGB").save(primary, format="JPEG", quality=85)
        _save_multi_camera_frame_images(row, episode_id, fi)


def _find_catalog_moment(episode_id: int, frame_index: int) -> dict | None:
    return catalog.get_moment(int(episode_id), int(frame_index))


def _catalog_moments_for_episode(episode_id: int) -> list[dict]:
    return catalog.get_episode_moments(int(episode_id))


def _compute_episode_score_trace(
    episode_id: int,
    dataset_id: str = loader.DROID_DATASET,
    preferred_image_key: str = "observation.images.wrist_left",
) -> list[dict]:
    catalog.set_episode_score_state(episode_id, "scoring", path=CATALOG_PATH)
    episode = _load_episode_for_query(dataset_id=dataset_id, episode_id=episode_id)
    if not episode:
        catalog.set_episode_score_state(episode_id, "failed", "episode not found", CATALOG_PATH)
        return []
    image_key = _resolve_image_key(episode[0], preferred_image_key)
    if image_key is None:
        catalog.set_episode_score_state(episode_id, "failed", "no image key resolved", CATALOG_PATH)
        return []
    frame_indices = [int(frame["frame_index"]) for frame in episode]
    camera_keys = sorted(
        {
            key
            for frame in episode
            for key in _frame_camera_keys(frame)
        }
    )
    if not camera_keys:
        catalog.set_episode_score_state(episode_id, "failed", "no camera keys available", CATALOG_PATH)
        return []
    images_by_camera = {
        key: [
            _as_pil_image(frame[key]).convert("RGB") if frame.get(key) is not None else None
            for frame in episode
        ]
        for key in camera_keys
    }
    scorer = FrameScorer(contamination=0.05, batch_size=32)
    bad_indices, cluster_labels, _, anomaly_scores, per_camera = scorer.flag_multiview(
        images_by_camera=images_by_camera,
        frame_indices=frame_indices,
        primary_camera=image_key,
    )
    bad_set = set(int(i) for i in bad_indices)
    trace = [
        {
            "frame_index": int(frame_indices[i]),
            "score": float(anomaly_scores[i]),
            "cluster_id": int(cluster_labels[i]),
            "flagged": bool(i in bad_set),
            "camera_scores": {
                _camera_tag(cam): float(cam_data["scores"][i])
                for cam, cam_data in per_camera.items()
            },
            "camera_flags": {
                _camera_tag(cam): bool(cam_data["flagged"][i])
                for cam, cam_data in per_camera.items()
            },
            "camera_clusters": {
                _camera_tag(cam): (
                    None if cam_data["clusters"][i] is None else int(cam_data["clusters"][i])
                )
                for cam, cam_data in per_camera.items()
            },
        }
        for i in range(len(frame_indices))
    ]
    catalog.save_episode_scores(episode_id=episode_id, score_trace=trace, path=CATALOG_PATH)
    if episode_id in catalog.get_scanned_episode_ids(CATALOG_PATH):
        flagged_count = sum(1 for row in trace if bool(row["flagged"]))
        catalog.mark_episode_scanned(
            episode_id=episode_id,
            frame_count=len(frame_indices),
            flagged_count=flagged_count,
            path=CATALOG_PATH,
        )
    _episode_scores[episode_id] = trace
    return trace


def _episode_clip_catalog(episode_id: int) -> list[dict]:
    moments = _catalog_moments_for_episode(episode_id)
    if not moments:
        return []
    clips_by_key: dict[str, dict] = {}
    for m in moments:
        frame_index = int(m.get("frame_index", 0))
        clip_frames_raw = m.get("clip_frame_indices")
        if isinstance(clip_frames_raw, list) and clip_frames_raw:
            clip_frames = sorted({int(fi) for fi in clip_frames_raw})
        else:
            context = [int(fi) for fi in m.get("context_window", [])]
            clip_frames = sorted(set(context + [frame_index]))
        if not clip_frames:
            continue
        start_f = clip_frames[0]
        end_f = clip_frames[-1]
        key = f"{start_f}:{end_f}"
        if key not in clips_by_key:
            clips_by_key[key] = {
                "id": "",
                "start_frame_index": start_f,
                "end_frame_index": end_f,
                "frame_indices": clip_frames,
                "anchor_frame_index": frame_index,
                "anchor_timestamp_s": m.get("clip_anchor_timestamp_s"),
                "start_timestamp_s": m.get("clip_start_timestamp_s"),
                "end_timestamp_s": m.get("clip_end_timestamp_s"),
                "cluster_id": int(m.get("cluster_id", -1)),
                "moment_count": 0,
            }
        clips_by_key[key]["moment_count"] += 1
        # Keep the earliest anchor for deterministic referencing.
        if frame_index < int(clips_by_key[key]["anchor_frame_index"]):
            clips_by_key[key]["anchor_frame_index"] = frame_index

    clips = sorted(clips_by_key.values(), key=lambda c: (int(c["start_frame_index"]), int(c["end_frame_index"])))
    for i, clip in enumerate(clips, start=1):
        clip["id"] = f"clip_{i}"
        clip["ordinal"] = i
    return clips


def _extract_clip_ref_ordinal(ref: str) -> int | None:
    text = ref.lower()
    match = re.search(r"\b(\d+)\s*(st|nd|rd|th)?\s+marked\s+clip\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\bclip\s*(\d+)\b", text)
    if match:
        return int(match.group(1))
    for word, value in ORDINAL_WORDS.items():
        if f"{word} marked clip" in text or f"{word} clip" in text:
            return value
    return None


def _resolve_clip_reference(episode_id: int, ref: str) -> dict | None:
    clips = _episode_clip_catalog(episode_id)
    if not clips:
        return None
    text = ref.strip().lower()
    # Direct id reference: clip_3
    direct = next((c for c in clips if c["id"] == text), None)
    if direct is not None:
        return direct
    ordinal = _extract_clip_ref_ordinal(text)
    if ordinal is not None:
        for c in clips:
            if int(c["ordinal"]) == ordinal:
                return c
    return None


def _downsample_ordered_frames(frames: list[dict], max_frames: int | None) -> list[dict]:
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    selected: list[dict] = []
    n = len(frames)
    for i in range(max_frames):
        pos = round(i * (n - 1) / (max_frames - 1))
        selected.append(frames[pos])
    return selected


def _estimate_tokens(text: str) -> int:
    # Lightweight heuristic for context budgeting.
    return max(1, len(text) // 4)


def _estimate_history_tokens(history: list[dict[str, str]]) -> int:
    total = 0
    for msg in history:
        total += _estimate_tokens(f"{msg.get('role', '')}: {msg.get('content', '')}")
    return total


def _compress_chat_history(
    history: list[dict[str, str]],
    max_tokens: int,
) -> tuple[list[dict[str, str]], bool, int]:
    cleaned = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in history
        if str(m.get("content", "")).strip()
    ]
    if not cleaned:
        return [], False, 0
    cur_tokens = _estimate_history_tokens(cleaned)
    if cur_tokens <= max_tokens:
        return cleaned, False, cur_tokens

    target_tail = max(1, int(max_tokens * 0.6))
    kept: list[dict[str, str]] = []
    used = 0
    for msg in reversed(cleaned):
        t = _estimate_tokens(f"{msg['role']}: {msg['content']}")
        if used + t > target_tail and kept:
            break
        kept.append(msg)
        used += t
    kept = list(reversed(kept))
    dropped = cleaned[: max(0, len(cleaned) - len(kept))]
    summary_lines = []
    for msg in dropped[-12:]:
        role = "u" if msg["role"] == "user" else "a"
        summary_lines.append(f"{role}:{msg['content'][:120]}")
    summary = " | ".join(summary_lines)[:1200]
    compacted = [{"role": "assistant", "content": f"memory summary: {summary}"}] + kept if summary else kept

    # Final hard trim if still above max budget.
    while _estimate_history_tokens(compacted) > max_tokens and len(compacted) > 1:
        compacted.pop(1)
    return compacted, True, _estimate_history_tokens(compacted)


def _build_clip_frames_payload(
    episode_id: int,
    frame_index: int,
    dataset_id: str = loader.DROID_DATASET,
    camera: str | None = None,
    max_frames: int | None = None,
) -> dict:
    moment = _find_catalog_moment(episode_id, frame_index)
    if moment is None:
        raise HTTPException(status_code=404, detail="Moment not found in catalog")

    clip_frame_indices = moment.get("clip_frame_indices")
    if not isinstance(clip_frame_indices, list) or not clip_frame_indices:
        context = [int(fi) for fi in moment.get("context_window", [])]
        clip_frame_indices = sorted(set(context + [int(frame_index)]))
    clip_frame_indices = [int(fi) for fi in clip_frame_indices]
    if not clip_frame_indices:
        raise HTTPException(status_code=400, detail="No clip frames available for this moment")

    episode = _load_episode_for_query(dataset_id=dataset_id, episode_id=episode_id)
    _materialize_frame_files(episode, episode_id=episode_id, frame_indices=clip_frame_indices)
    fps = loader.dataset_fps(dataset_id)
    ts_map = {
        int(row.get("frame_index", i)): float(ts)
        for i, (row, ts) in enumerate(zip(episode, frame_timestamps(episode, fps=fps)))
    }

    row_map = {int(row.get("frame_index", -1)): row for row in episode}
    ordered_frames: list[dict] = []
    normalized_camera = _camera_tag(_camera_key_from_tag_or_key(camera)) if camera else None
    for fi in clip_frame_indices:
        if fi not in row_map:
            continue
        if normalized_camera is None:
            views = _frame_views_payload(episode_id, fi)
        else:
            views = []
            tagged = FRAMES_DIR / f"{episode_id}_{fi}__{normalized_camera}.jpg"
            if tagged.exists():
                views = [{"camera": normalized_camera, "frame_url": f"/api/frames/{tagged.name}"}]
            else:
                primary = FRAMES_DIR / f"{episode_id}_{fi}.jpg"
                if primary.exists():
                    views = [{"camera": "primary", "frame_url": f"/api/frames/{primary.name}"}]
        if not views:
            continue
        ordered_frames.append(
            {
                "frame_index": fi,
                "timestamp_s": ts_map.get(fi, fi / max(0.001, fps)),
                "views": views,
            }
        )

    if not ordered_frames:
        raise HTTPException(status_code=404, detail="Clip frames could not be materialized")
    ordered_frames = _downsample_ordered_frames(ordered_frames, max_frames=max_frames)

    return {
        "episode_id": int(episode_id),
        "frame_index": int(frame_index),
        "camera": normalized_camera,
        "clip_pre_s": moment.get("clip_pre_s"),
        "clip_post_s": moment.get("clip_post_s"),
        "clip_fps": moment.get("clip_fps", fps),
        "clip_strategy": moment.get("clip_strategy"),
        "clip_anchor_timestamp_s": moment.get("clip_anchor_timestamp_s"),
        "clip_start_timestamp_s": moment.get("clip_start_timestamp_s"),
        "clip_end_timestamp_s": moment.get("clip_end_timestamp_s"),
        "frames": ordered_frames,
    }


def _save_multi_camera_frame_images(frame: dict, episode_id: int, frame_index: int) -> None:
    for key, value in frame.items():
        if not key.startswith("observation.images.") or value is None:
            continue
        tag = _camera_tag(key)
        filename = f"{episode_id}_{frame_index}__{tag}.jpg"
        out = FRAMES_DIR / filename
        if out.exists():
            continue
        _as_pil_image(value).convert("RGB").save(out, format="JPEG", quality=82)


def _video_file_is_usable(path: Path, min_frames: int = 2) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frame_count = 0
            for _ in container.decode(stream):
                frame_count += 1
                if frame_count >= min_frames:
                    return True
        return False
    except Exception:
        return False


def _ensure_episode_media_files(
    episode_id: int,
    dataset_id: str = loader.DROID_DATASET,
    preferred_image_key: str = "observation.images.wrist_left",
) -> tuple[str | None, dict[str, str]]:
    primary_path = FRAMES_DIR / f"episode_{episode_id}.mp4"
    inferred_camera_paths = sorted(FRAMES_DIR.glob(f"episode_{episode_id}__*.mp4"))
    primary_ok = _video_file_is_usable(primary_path)
    camera_ok = all(_video_file_is_usable(path) for path in inferred_camera_paths) if inferred_camera_paths else False
    if primary_ok and camera_ok:
        return (
            f"/api/videos/{primary_path.name}" if primary_path.exists() else None,
            {path.stem.split('__', 1)[-1]: f"/api/videos/{path.name}" for path in inferred_camera_paths},
        )

    for path in [primary_path, *inferred_camera_paths]:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    episode = _load_episode_for_query(dataset_id=dataset_id, episode_id=episode_id)
    camera_videos = _save_episode_camera_videos(episode, episode_id)
    image_key = _resolve_image_key(episode[0], preferred_image_key) or preferred_image_key
    primary_valid = [frame for frame in episode if frame.get(image_key) is not None]
    video_url = camera_videos.get(_camera_tag(image_key)) or _save_episode_video(primary_valid, image_key, episode_id)
    catalog.save_episode_media(
        episode_id=episode_id,
        video_url=video_url,
        camera_videos=camera_videos,
        path=CATALOG_PATH,
    )
    if video_url:
        _episode_videos[episode_id] = video_url
    if camera_videos:
        _episode_camera_videos[episode_id] = camera_videos
    return video_url, camera_videos


def _save_episode_video(
    episode: list[dict],
    image_key: str,
    episode_id: int,
    fps: int = 8,
) -> str | None:
    if not episode:
        return None
    out_name = f"episode_{episode_id}.mp4"
    out_path = FRAMES_DIR / out_name
    if out_path.exists() and _video_file_is_usable(out_path):
        return f"/api/videos/{out_name}"
    if out_path.exists():
        try:
            out_path.unlink()
        except OSError:
            pass

    import av
    import numpy as np

    first = _as_pil_image(episode[0][image_key]).convert("RGB")
    w = first.width - (first.width % 2)
    h = first.height - (first.height % 2)
    if w <= 0 or h <= 0:
        return None

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=max(1, fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast", "crf": "30"}

    try:
        for frame in episode:
            img = _as_pil_image(frame[image_key]).convert("RGB")
            if img.width != w or img.height != h:
                img = img.resize((w, h))
            arr = np.asarray(img, dtype=np.uint8)
            video_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    return f"/api/videos/{out_name}"


def _save_episode_camera_videos(
    episode: list[dict],
    episode_id: int,
    fps: int = 8,
) -> dict[str, str]:
    if not episode:
        return {}
    camera_keys = sorted(
        {
            key
            for frame in episode
            for key, value in frame.items()
            if key.startswith("observation.images.") and value is not None
        }
    )
    if not camera_keys:
        return {}

    import av
    import numpy as np

    out: dict[str, str] = {}
    for key in camera_keys:
        tag = _camera_tag(key)
        out_name = f"episode_{episode_id}__{tag}.mp4"
        out_path = FRAMES_DIR / out_name
        if out_path.exists() and _video_file_is_usable(out_path):
            out[tag] = f"/api/videos/{out_name}"
            continue
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass

        first_idx = next((i for i, frame in enumerate(episode) if frame.get(key) is not None), None)
        if first_idx is None:
            continue
        first = _as_pil_image(episode[first_idx][key]).convert("RGB")
        w = first.width - (first.width % 2)
        h = first.height - (first.height % 2)
        if w <= 0 or h <= 0:
            continue

        container = av.open(str(out_path), mode="w")
        stream = container.add_stream("libx264", rate=max(1, fps))
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "30"}

        last_img = first
        try:
            for frame in episode:
                value = frame.get(key)
                if value is not None:
                    last_img = _as_pil_image(value).convert("RGB")
                img = last_img
                if img.width != w or img.height != h:
                    img = img.resize((w, h))
                arr = np.asarray(img, dtype=np.uint8)
                video_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

        out[tag] = f"/api/videos/{out_name}"
    return out


@app.get("/api/frame-views/{episode_id}/{frame_index}")
def get_frame_views(episode_id: int, frame_index: int):
    return _frame_views_payload(episode_id, frame_index)


@app.get("/api/clip-frames/{episode_id}/{frame_index}")
def get_clip_frames(
    episode_id: int,
    frame_index: int,
    dataset_id: str = loader.DROID_DATASET,
    camera: str | None = None,
):
    """Materialize deterministic clip frames for query-time VLM calls.

    Uses persisted clip metadata when available; falls back to context_window.
    """
    return _build_clip_frames_payload(
        episode_id=episode_id,
        frame_index=frame_index,
        dataset_id=dataset_id,
        camera=camera,
    )


@app.get("/api/episode-clips/{episode_id}")
def get_episode_clips(episode_id: int):
    return _episode_clip_catalog(episode_id)


@app.get("/api/clips/resolve/{episode_id}")
def resolve_clip_reference(episode_id: int, ref: str):
    clip = _resolve_clip_reference(episode_id=episode_id, ref=ref)
    if clip is None:
        raise HTTPException(status_code=404, detail="Could not resolve clip reference")
    return clip


class VlmQueryRequest(BaseModel):
    episode_id: int
    frame_index: int
    dataset_id: str = loader.DROID_DATASET
    camera: str | None = None
    question: str | None = None
    target_clip_id: str | None = None
    resolve_clip_refs: bool = True
    model_id: str = VLM_MODEL_ID
    max_frames: int = 12
    max_retries: int = 2
    temperature: float = 0.0
    use_tools: bool = True
    use_cache: bool = True
    chat_history: list[dict[str, str]] = []
    max_context_tokens: int = 1200


@app.post("/api/vlm/query")
def vlm_query(payload: VlmQueryRequest):
    resolved_clip: dict | None = None
    if payload.resolve_clip_refs and payload.question:
        resolved_clip = _resolve_clip_reference(payload.episode_id, payload.question)
    if resolved_clip is None and payload.target_clip_id:
        resolved_clip = _resolve_clip_reference(payload.episode_id, payload.target_clip_id)

    target_frame_index = payload.frame_index
    if resolved_clip is not None:
        target_frame_index = int(resolved_clip.get("anchor_frame_index", payload.frame_index))

    compacted_history, history_compacted, history_tokens = _compress_chat_history(
        history=payload.chat_history[-40:],
        max_tokens=max(200, int(payload.max_context_tokens)),
    )
    clip_payload = _build_clip_frames_payload(
        episode_id=payload.episode_id,
        frame_index=target_frame_index,
        dataset_id=payload.dataset_id,
        camera=payload.camera,
        max_frames=max(1, payload.max_frames),
    )

    # Build clip-identity metadata so the VLM knows which incident it's analysing
    # and how it sits relative to the other flagged clips in the episode.
    all_episode_clips = _episode_clip_catalog(payload.episode_id)
    if resolved_clip is not None:
        siblings = [c for c in all_episode_clips if c["id"] != resolved_clip["id"]]
        clip_metadata: dict | None = {
            **resolved_clip,
            "total_clips": len(all_episode_clips),
            "sibling_clips": siblings,
        }
    elif all_episode_clips:
        # No named clip resolved — still inject frame-level context from clip_payload
        clip_metadata = {
            "total_clips": len(all_episode_clips),
            "anchor_timestamp_s": clip_payload.get("clip_anchor_timestamp_s"),
            "start_timestamp_s": clip_payload.get("clip_start_timestamp_s"),
            "end_timestamp_s": clip_payload.get("clip_end_timestamp_s"),
            "sibling_clips": all_episode_clips,
        }
    else:
        clip_metadata = None
    cache_key = json.dumps(
        {
            "episode_id": payload.episode_id,
            "frame_index": payload.frame_index,
            "dataset_id": payload.dataset_id,
            "camera": payload.camera,
            "question": payload.question,
            "target_clip_id": payload.target_clip_id,
            "resolved_clip_id": (resolved_clip or {}).get("id"),
            "resolve_clip_refs": bool(payload.resolve_clip_refs),
            "model_id": payload.model_id,
            "max_frames": max(1, payload.max_frames),
            "max_retries": max(1, payload.max_retries),
            "temperature": payload.temperature,
            "use_tools": bool(payload.use_tools),
            "chat_history": compacted_history,
            "max_context_tokens": max(200, int(payload.max_context_tokens)),
            "clip_frames": [f["frame_index"] for f in clip_payload["frames"]],
        },
        sort_keys=True,
    )
    if payload.use_cache and cache_key in _vlm_query_cache:
        return {"cached": True, **_vlm_query_cache[cache_key]}

    images: list[Image.Image] = []
    frame_indices: list[int] = []
    timestamps_s: list[float] = []
    for frame in clip_payload["frames"]:
        views = frame.get("views", [])
        if not views:
            continue
        frame_url = views[0].get("frame_url")
        if not isinstance(frame_url, str):
            continue
        filename = frame_url.rsplit("/", 1)[-1]
        image_path = FRAMES_DIR / filename
        if not image_path.exists():
            continue
        with Image.open(image_path) as opened:
            images.append(opened.convert("RGB"))
        frame_indices.append(int(frame["frame_index"]))
        timestamps_s.append(float(frame["timestamp_s"]))

    if not images:
        raise HTTPException(status_code=404, detail="No clip images available for VLM query")

    try:
        result = query_incident(
            images=images,
            frame_indices=frame_indices,
            timestamps_s=timestamps_s,
            chat_history=compacted_history,
            user_query=payload.question,
            clip_metadata=clip_metadata,
            model_id=payload.model_id,
            max_retries=max(1, payload.max_retries),
            temperature=payload.temperature,
            use_tools=bool(payload.use_tools),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"VLM query failed: {exc}") from exc

    _vlm_token_totals["input_tokens"] += int(result.input_tokens)
    _vlm_token_totals["output_tokens"] += int(result.output_tokens)
    _vlm_token_totals["total_tokens"] = _vlm_token_totals["input_tokens"] + _vlm_token_totals["output_tokens"]
    print(
        json.dumps(
            {
                "component": "vlm",
                "event": "query",
                "episode_id": payload.episode_id,
                "frame_index": target_frame_index,
                "resolved_clip_id": (resolved_clip or {}).get("id"),
                "model": payload.model_id,
                "attempts": result.attempts,
                "history_tokens_est": history_tokens,
                "history_compacted": history_compacted,
                "input_tokens": int(result.input_tokens),
                "output_tokens": int(result.output_tokens),
                "accumulated_total_tokens": int(_vlm_token_totals["total_tokens"]),
            }
        )
    )

    response = {
        "clip": clip_payload,
        "resolved_clip": resolved_clip,
        "episode_clips": all_episode_clips,
        "model": payload.model_id,
        "attempts": result.attempts,
        "result": result.result.model_dump(),
        "raw_text": result.raw_text,
        "token_usage": {
            "input_tokens": int(result.input_tokens),
            "output_tokens": int(result.output_tokens),
            "total_tokens": int(result.input_tokens + result.output_tokens),
            "accumulated_total_tokens": int(_vlm_token_totals["total_tokens"]),
        },
        "context": {
            "history_tokens_est": history_tokens,
            "history_compacted": history_compacted,
            "history_messages": len(compacted_history),
            "context_token_limit": max(200, int(payload.max_context_tokens)),
        },
    }
    if payload.use_cache:
        _vlm_query_cache[cache_key] = response
    return {"cached": False, **response}


def _run_scan(params: ScanParams) -> None:
    global _scan_thread
    producer: threading.Thread | None = None
    stop_event = threading.Event()
    current_episode: int | None = None
    try:
        episode_fps = loader.dataset_fps(params.dataset_id)
        scorer = FrameScorer(
            contamination=params.contamination,
            batch_size=params.embed_batch_size,
        )
        _set_scan_telemetry(
            active=True,
            paused=False,
            started_at=time.time(),
            queue_size=0,
            queue_capacity=1,
            current_episode=None,
            loaded_episodes=0,
            scored_episodes=0,
            loaded_frames=0,
            scored_frames=0,
            total_flagged=0,
        )
        total_flagged = 0
        max_frames = None if params.max_frames_per_episode <= 0 else int(params.max_frames_per_episode)
        work_q: queue.Queue[object] = queue.Queue(maxsize=1)
        producer_errors: list[Exception] = []
        sentinel = object()

        def _producer() -> None:
            logged_keys = False
            already_scanned = catalog.get_scanned_episode_ids()
            queued_new_episodes = 0
            if already_scanned:
                print(f"[droid-loop] resuming — {len(already_scanned)} episodes already scanned, skipping")
            effective_start_episode = int(params.start_episode)
            while effective_start_episode in already_scanned:
                effective_start_episode += 1
            try:
                for episode in loader.stream_episodes(
                    params.dataset_id,
                    image_keys=[
                        params.image_key,
                        "observation.images.wrist_left",
                        "observation.images.exterior_1_left",
                        "observation.images.exterior_2_left",
                    ],
                    start_episode=effective_start_episode,
                    max_episodes=None,
                ):
                    if _scan_should_stop(stop_event):
                        return
                    if (
                        params.max_episodes is not None
                        and queued_new_episodes >= params.max_episodes
                    ):
                        return
                    _wait_if_scan_paused()
                    if _scan_should_stop(stop_event):
                        return
                    episode_id = episode[0]["episode_index"]

                    # Resume: skip episodes we've already fully scored
                    if episode_id in already_scanned:
                        _log_scan_event("episode_skipped_resume", episode_id=episode_id)
                        continue

                    _log_scan_event(
                        "episode_loaded",
                        episode_id=episode_id,
                        episode_frames=len(episode),
                        rss_mb=_rss_mb(),
                    )
                    _set_scan_telemetry(current_episode=episode_id)

                    if _rss_mb() > params.max_rss_mb:
                        raise RuntimeError(
                            f"memory guard triggered before queueing ep {episode_id}: "
                            f"rss_mb={_rss_mb()} > max_rss_mb={params.max_rss_mb}"
                        )

                    if not logged_keys:
                        logged_keys = True
                        all_keys = list(episode[0].keys())
                        img_keys = _find_image_keys(episode[0])
                        print(f"[scan] frame keys: {all_keys}")
                        print(f"[scan] detected image keys: {img_keys}")

                    image_key = _resolve_image_key(episode[0], params.image_key)
                    if image_key is None:
                        _push(
                            {
                                "type": "progress",
                                "episode_id": episode_id,
                                "frames": len(episode),
                                "flagged": 0,
                                "total_flagged": total_flagged,
                                "note": "no image key found, skipped",
                            }
                        )
                        continue

                    camera_keys = sorted(
                        {
                            key
                            for frame in episode
                            for key in _frame_camera_keys(frame)
                        }
                    )
                    if not camera_keys:
                        continue
                    if image_key not in camera_keys:
                        image_key = camera_keys[0]
                    valid = [f for f in episode if any(f.get(k) is not None for k in camera_keys)]
                    if not valid:
                        continue
                    if max_frames is not None and len(valid) > max_frames:
                        step = max(1, len(valid) // max_frames)
                        valid = valid[::step][:max_frames]
                        _log_scan_event(
                            "episode_downsampled",
                            episode_id=episode_id,
                            selected_frames=len(valid),
                            step=step,
                        )

                    frame_indices = [f["frame_index"] for f in valid]
                    with _scan_telemetry_lock:
                        _scan_telemetry["loaded_episodes"] = int(_scan_telemetry["loaded_episodes"]) + 1
                        _scan_telemetry["loaded_frames"] = int(_scan_telemetry["loaded_frames"]) + len(valid)
                    camera_videos = _save_episode_camera_videos(valid, episode_id)
                    if camera_videos:
                        _episode_camera_videos[episode_id] = camera_videos
                    primary_tag = _camera_tag(image_key)
                    primary_valid = [f for f in valid if f.get(image_key) is not None]
                    video_url = camera_videos.get(primary_tag) or _save_episode_video(
                        primary_valid, image_key, episode_id
                    )
                    if video_url:
                        _episode_videos[episode_id] = video_url
                    catalog.save_episode_media(
                        episode_id=episode_id,
                        video_url=video_url,
                        camera_videos=camera_videos,
                        path=CATALOG_PATH,
                    )

                    # Save preview frames from ALL cameras per timestep so the UI
                    # can show all camera angles in sync during scan playback.
                    step = 1  # stream every frame as a preview
                    preview_groups: list[dict[str, str]] = []
                    print(
                        f"[scan] ep {episode_id}: {len(valid)} frames, saving multi-cam "
                        f"previews (step={step}, cameras={[_camera_tag(k) for k in camera_keys]}) "
                        f"to {FRAMES_DIR}"
                    )
                    for ep_i in range(0, len(valid), step):
                        fi = frame_indices[ep_i]
                        group: dict[str, str] = {}
                        for ck in camera_keys:
                            if valid[ep_i].get(ck) is None:
                                continue
                            cam_tag = _camera_tag(ck)
                            pname = f"prev_{episode_id}_{fi}_{cam_tag}.jpg"
                            try:
                                thumb = _as_pil_image(valid[ep_i][ck]).copy().convert("RGB")
                                thumb.thumbnail((480, 360))
                                thumb.save(FRAMES_DIR / pname, format="JPEG", quality=75)
                                group[cam_tag] = f"/api/frames/{pname}"
                            except Exception:
                                pass
                        if group:
                            preview_groups.append(group)

                    # Backward-compat flat list (primary camera only)
                    primary_tag = _camera_tag(image_key)
                    preview_urls = [
                        g.get(primary_tag) or next(iter(g.values()))
                        for g in preview_groups if g
                    ]

                    _push(
                        {
                            "type": "episode_frames",
                            "episode_id": episode_id,
                            "total_frames": len(valid),
                            "frame_urls": preview_urls,       # primary only (compat)
                            "preview_groups": preview_groups, # all cameras per step
                            "video_url": video_url,
                            "camera_videos": camera_videos,
                        }
                    )

                    payload = {
                        "episode_id": episode_id,
                        "episode": valid,
                        "image_key": image_key,
                        "frame_indices": frame_indices,
                    }
                    catalog.set_episode_score_state(episode_id, "pending", path=CATALOG_PATH)
                    while True:
                        if _scan_should_stop(stop_event):
                            return
                        _wait_if_scan_paused()
                        if _scan_should_stop(stop_event):
                            return
                        try:
                            work_q.put(payload, timeout=0.5)
                            _log_scan_event(
                                "episode_queued",
                                episode_id=episode_id,
                                queued_frames=len(valid),
                                queue_size=work_q.qsize(),
                                rss_mb=_rss_mb(),
                            )
                            _set_scan_telemetry(queue_size=work_q.qsize())
                            queued_new_episodes += 1
                            break
                        except queue.Full:
                            if _rss_mb() > params.max_rss_mb:
                                raise RuntimeError(
                                    f"memory guard triggered while queueing ep {episode_id}: "
                                    f"rss_mb={_rss_mb()} > max_rss_mb={params.max_rss_mb}"
                                )
                            continue
            except Exception as exc:
                producer_errors.append(exc)
            finally:
                while True:
                    if _scan_should_stop(stop_event):
                        break
                    try:
                        work_q.put(sentinel, timeout=0.5)
                        break
                    except queue.Full:
                        continue

        # Daemon thread ensures server shutdown is not blocked if upstream I/O stalls.
        producer = threading.Thread(target=_producer, name="scan-loader", daemon=True)
        producer.start()

        while True:
            if _scan_should_stop(stop_event):
                break
            _wait_if_scan_paused()
            item = work_q.get()
            if item is sentinel:
                break
            if not isinstance(item, dict):
                continue

            episode_id = int(item["episode_id"])
            current_episode = episode_id
            episode = item["episode"]
            image_key = str(item["image_key"])
            frame_indices = item["frame_indices"]
            catalog.set_episode_score_state(episode_id, "scoring", path=CATALOG_PATH)
            camera_keys = sorted(
                {
                    key
                    for frame in episode
                    for key in _frame_camera_keys(frame)
                }
            )
            images_by_camera = {
                key: [
                    _as_pil_image(frame[key]).convert("RGB") if frame.get(key) is not None else None
                    for frame in episode
                ]
                for key in camera_keys
            }
            _set_scan_telemetry(current_episode=episode_id, queue_size=work_q.qsize())
            _wait_if_scan_paused()

            bad_indices, cluster_labels, cluster_meta, anomaly_scores, per_camera = scorer.flag_multiview(
                images_by_camera=images_by_camera,
                frame_indices=frame_indices,
                primary_camera=image_key,
            )
            merged_bad_indices, incident_windows = _merge_flag_positions(
                bad_indices,
                frame_indices,
                anomaly_scores,
                merge_gap_frames=params.incident_merge_gap_frames,
            )
            bad_set = set(merged_bad_indices)
            _episode_scores[episode_id] = [
                {
                    "frame_index": int(frame_indices[i]),
                    "score": float(anomaly_scores[i]),
                    "cluster_id": int(cluster_labels[i]),
                    "flagged": bool(i in bad_set),
                    "camera_scores": {
                        _camera_tag(cam): float(cam_data["scores"][i])
                        for cam, cam_data in per_camera.items()
                    },
                    "camera_flags": {
                        _camera_tag(cam): bool(cam_data["flagged"][i])
                        for cam, cam_data in per_camera.items()
                    },
                    "camera_clusters": {
                        _camera_tag(cam): (
                            None if cam_data["clusters"][i] is None else int(cam_data["clusters"][i])
                        )
                        for cam, cam_data in per_camera.items()
                    },
                }
                for i in range(len(frame_indices))
            ]
            catalog.save_episode_scores(
                episode_id=episode_id,
                score_trace=_episode_scores[episode_id],
                path=CATALOG_PATH,
            )
            _log_scan_event(
                "episode_scored",
                episode_id=episode_id,
                scored_frames=len(frame_indices),
                flagged_frames=len(merged_bad_indices),
                raw_flagged_frames=len(bad_indices),
                incident_windows=len(incident_windows),
                queue_size=work_q.qsize(),
                rss_mb=_rss_mb(),
            )
            with _scan_telemetry_lock:
                _scan_telemetry["scored_episodes"] = int(_scan_telemetry["scored_episodes"]) + 1
                _scan_telemetry["scored_frames"] = int(_scan_telemetry["scored_frames"]) + len(frame_indices)

            if _rss_mb() > params.max_rss_mb:
                raise RuntimeError(
                    f"memory guard triggered after scoring ep {episode_id}: "
                    f"rss_mb={_rss_mb()} > max_rss_mb={params.max_rss_mb}"
                )

            window_by_anchor = {
                int(w["anchor_pos"]): (int(w["start_frame"]), int(w["end_frame"]))
                for w in incident_windows
            }
            for idx in merged_bad_indices:
                if _scan_should_stop(stop_event):
                    break
                _wait_if_scan_paused()
                frame = episode[idx]
                cl_id = int(cluster_labels[idx])
                meta = cluster_meta.get(cl_id, {})
                incident_start, incident_end = window_by_anchor.get(
                    int(idx),
                    (int(frame["frame_index"]), int(frame["frame_index"])),
                )
                lo = max(0, idx - params.context_window)
                hi = min(len(episode) - 1, idx + params.context_window)
                context = [episode[j]["frame_index"] for j in range(lo, hi + 1) if j != idx]
                clip, _ = extract_clip(
                    episode_rows=episode,
                    request=ClipRequest(
                        anchor_frame_index=int(frame["frame_index"]),
                        pre_s=2.0,
                        post_s=2.0,
                        max_frames=16,
                        strategy="uniform",
                    ),
                    fps=episode_fps,
                )

                frame_index = frame["frame_index"]
                key_for_frame = _resolve_preview_key(frame, image_key)
                if key_for_frame is None:
                    continue
                img = _as_pil_image(frame[key_for_frame]).convert("RGB")
                img_filename = f"{episode_id}_{frame_index}.jpg"
                img.save(FRAMES_DIR / img_filename, format="JPEG", quality=85)
                _save_multi_camera_frame_images(frame, episode_id, frame_index)

                for j in range(lo, hi + 1):
                    if j == idx:
                        continue
                    ctx_frame = episode[j]
                    ctx_frame_index = ctx_frame["frame_index"]
                    ctx_filename = f"{episode_id}_{ctx_frame_index}.jpg"
                    key_for_ctx = _resolve_preview_key(ctx_frame, image_key)
                    if not (FRAMES_DIR / ctx_filename).exists():
                        if key_for_ctx is not None:
                            _as_pil_image(ctx_frame[key_for_ctx]).convert("RGB").save(
                                FRAMES_DIR / ctx_filename, format="JPEG", quality=80
                            )
                    _save_multi_camera_frame_images(ctx_frame, episode_id, ctx_frame_index)

                catalog.save_moment(
                    {
                        "episode_id": episode_id,
                        "frame_index": frame_index,
                        "label": "anomaly",
                        "cluster_id": cl_id,
                        "cluster_size": meta.get("size", 0),
                        "cluster_span": meta.get("span", []),
                        "incident_span": [incident_start, incident_end],
                        "context_window": context,
                        "clip_pre_s": 2.0,
                        "clip_post_s": 2.0,
                        "clip_fps": episode_fps,
                        "clip_strategy": "uniform",
                        "clip_frame_indices": [f.frame_index for f in clip.frames],
                        "clip_anchor_timestamp_s": clip.anchor_timestamp_s,
                        "clip_start_timestamp_s": clip.start_timestamp_s,
                        "clip_end_timestamp_s": clip.end_timestamp_s,
                    },
                    CATALOG_PATH,
                )

                _push(
                    {
                        "type": "frame",
                        "episode_id": episode_id,
                        "frame_index": frame_index,
                        "cluster_id": cl_id,
                        "cluster_size": meta.get("size", 0),
                        "incident_span": [incident_start, incident_end],
                        "frame_url": f"/api/frames/{img_filename}",
                        "total_flagged": total_flagged + 1,
                    }
                )

            total_flagged += len(merged_bad_indices)
            _set_scan_telemetry(total_flagged=total_flagged, queue_size=work_q.qsize())
            _push(
                {
                    "type": "progress",
                    "episode_id": episode_id,
                    "frames": len(episode),
                    "flagged": len(merged_bad_indices),
                    "total_flagged": total_flagged,
                }
            )
            catalog.mark_episode_scanned(
                episode_id,
                frame_count=len(episode),
                flagged_count=len(merged_bad_indices),
            )
            _log_scan_event(
                "episode_done",
                episode_id=episode_id,
                total_flagged=total_flagged,
                queue_size=work_q.qsize(),
                rss_mb=_rss_mb(),
            )

        stop_event.set()
        producer.join(timeout=5.0)
        if producer_errors:
            raise producer_errors[0]

        if not _scan_should_stop(stop_event):
            _push({"type": "done", "total_flagged": total_flagged})

    except Exception as e:
        if current_episode is not None:
            try:
                catalog.set_episode_score_state(int(current_episode), "failed", str(e), CATALOG_PATH)
            except Exception:
                pass
        if not _scan_should_stop(stop_event):
            _push({"type": "error", "message": str(e)})
    finally:
        stop_event.set()
        # Ensure paused scans can unwind and producer can exit.
        _scan_pause_event.set()
        if producer is not None and producer.is_alive():
            producer.join(timeout=5.0)
        _set_scan_telemetry(active=False, paused=False, current_episode=None, queue_size=0)
        with _job_lock:
            _job_running["name"] = None
        if _scan_thread is threading.current_thread():
            _scan_thread = None


@app.post("/api/scan")
def start_scan(params: ScanParams):
    global _scan_thread
    if _server_stop_event.is_set():
        raise HTTPException(503, "server is stopping")
    with _job_lock:
        if _job_running["name"]:
            raise HTTPException(409, f"'{_job_running['name']}' is already running")
        if _scan_thread is not None and _scan_thread.is_alive():
            raise HTTPException(409, "scan worker is still shutting down")
        _job_running["name"] = "scan"
        _scan_cancel_event.clear()
        _scan_pause_event.set()
        _set_scan_telemetry(
            active=True,
            paused=False,
            started_at=time.time(),
            queue_size=0,
            queue_capacity=1,
            current_episode=None,
            loaded_episodes=0,
            scored_episodes=0,
            loaded_frames=0,
            scored_frames=0,
            total_flagged=0,
        )
        _episode_scores.clear()
        _episode_videos.clear()
        _episode_camera_videos.clear()
        _clear_event_queue()
    _scan_thread = threading.Thread(target=_run_scan, args=(params,), name="scan-main", daemon=True)
    _scan_thread.start()
    return {"status": "started"}


@app.post("/api/scan/pause")
def pause_scan():
    with _job_lock:
        if _job_running["name"] != "scan":
            raise HTTPException(409, "scan is not running")
    _scan_pause_event.clear()
    _set_scan_telemetry(paused=True)
    return {"status": "paused"}


@app.post("/api/scan/resume")
def resume_scan():
    with _job_lock:
        if _job_running["name"] != "scan":
            raise HTTPException(409, "scan is not running")
    _scan_pause_event.set()
    _set_scan_telemetry(paused=False)
    return {"status": "running"}


@app.get("/api/scan/events")
def scan_events():
    return StreamingResponse(_sse_stream(), media_type="text/event-stream")


# ── Label ─────────────────────────────────────────────────────────────────────

class LabelParams(BaseModel):
    dataset_id: str = loader.DROID_DATASET
    image_key: str = "observation.images.wrist_left"
    n_samples: int = 3


def _run_label(params: LabelParams) -> None:
    global _label_thread
    if _server_stop_event.is_set() or _label_cancel_event.is_set():
        return
    try:
        if _label_cancel_event.is_set():
            return
        descriptions = labeler.label_clusters(
            CATALOG_PATH,
            params.dataset_id,
            params.image_key,
            DESCRIPTIONS_PATH,
            params.n_samples,
        )
        total = sum(len(v) for v in descriptions.values())
        if not _server_stop_event.is_set() and not _label_cancel_event.is_set():
            _push({"type": "done", "total_clusters": total})
    except Exception as e:
        if not _server_stop_event.is_set() and not _label_cancel_event.is_set():
            _push({"type": "error", "message": str(e)})
    finally:
        with _job_lock:
            _job_running["name"] = None
        if _label_thread is threading.current_thread():
            _label_thread = None


@app.post("/api/label")
def start_label(params: LabelParams):
    global _label_thread
    if _server_stop_event.is_set():
        raise HTTPException(503, "server is stopping")
    with _job_lock:
        if _job_running["name"]:
            raise HTTPException(409, f"'{_job_running['name']}' is already running")
        if _label_thread is not None and _label_thread.is_alive():
            raise HTTPException(409, "label worker is still shutting down")
        if not CATALOG_PATH.exists():
            raise HTTPException(400, "No catalog — run scan first")
        _job_running["name"] = "label"
        _label_cancel_event.clear()
        _clear_event_queue()
    _label_thread = threading.Thread(target=_run_label, args=(params,), name="label-main", daemon=True)
    _label_thread.start()
    return {"status": "started"}


@app.get("/api/label/events")
def label_events():
    return StreamingResponse(_sse_stream(), media_type="text/event-stream")


# ── Export ────────────────────────────────────────────────────────────────────

@app.post("/api/export")
def do_export():
    if not CATALOG_PATH.exists():
        raise HTTPException(400, "No catalog — run scan first")
    signal.export(CATALOG_PATH, SIGNAL_PATH)
    moments = list(catalog.load_moments(CATALOG_PATH))
    return {"status": "ok", "exported": len(moments), "path": str(SIGNAL_PATH)}


@app.on_event("shutdown")
def on_shutdown() -> None:
    _server_stop_event.set()
    _scan_cancel_event.set()
    _label_cancel_event.set()
    _scan_pause_event.set()
    with _job_lock:
        _job_running["name"] = None
    _join_worker_threads(timeout_s=1.5)
    _wait_for_active_job_stop(timeout_s=1.0)
    catalog.close_all()
    _cleanup_runtime_state()
    _cleanup_artifacts()
