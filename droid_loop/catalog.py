"""
SQLite-backed catalog.

Provides:
  - O(1) lookups by (episode_id, frame_index)
  - resume support via scanned episode tracking
  - atomic deduplicated writes with INSERT OR REPLACE

Thread safety: WAL mode + per-thread connections via threading.local().
"""
import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "catalog.db"

# Keep the exported name stable for callers that import DEFAULT_CATALOG.
DEFAULT_CATALOG = DB_PATH

_local = threading.local()
_registry_lock = threading.Lock()
_all_conns: dict[str, sqlite3.Connection] = {}


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS moments (
    episode_id      INTEGER NOT NULL,
    frame_index     INTEGER NOT NULL,
    cluster_id      INTEGER,
    cluster_size    INTEGER,
    label           TEXT,
    verdict         TEXT,
    cluster_span    TEXT,     -- JSON array
    incident_span   TEXT,     -- JSON array
    context_window  TEXT,     -- JSON array
    clip_pre_s      REAL,
    clip_post_s     REAL,
    clip_fps        REAL,
    clip_strategy   TEXT,
    clip_frame_indices TEXT,  -- JSON array
    clip_anchor_timestamp_s REAL,
    clip_start_timestamp_s  REAL,
    clip_end_timestamp_s    REAL,
    extra           TEXT,     -- JSON blob for any future fields
    created_at      REAL DEFAULT (unixepoch('now', 'subsec')),
    PRIMARY KEY (episode_id, frame_index)
);

CREATE TABLE IF NOT EXISTS scanned_episodes (
    episode_id    INTEGER PRIMARY KEY,
    frame_count   INTEGER DEFAULT 0,
    flagged_count INTEGER DEFAULT 0,
    scanned_at    REAL DEFAULT (unixepoch('now', 'subsec'))
);

CREATE TABLE IF NOT EXISTS episode_scores (
    episode_id    INTEGER PRIMARY KEY,
    score_trace   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    updated_at    REAL DEFAULT (unixepoch('now', 'subsec'))
);

CREATE TABLE IF NOT EXISTS episode_media (
    episode_id    INTEGER PRIMARY KEY,
    video_url     TEXT,
    camera_videos TEXT,
    updated_at    REAL DEFAULT (unixepoch('now', 'subsec'))
);
"""

_KNOWN_COLUMNS = {
    "episode_id", "frame_index", "cluster_id", "cluster_size",
    "label", "verdict", "cluster_span", "incident_span", "context_window",
    "clip_pre_s", "clip_post_s", "clip_fps", "clip_strategy",
    "clip_frame_indices", "clip_anchor_timestamp_s",
    "clip_start_timestamp_s", "clip_end_timestamp_s",
}


# ── Connection ────────────────────────────────────────────────────────────────

def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    _ensure_runtime_schema(conn)
    return conn


def _ensure_runtime_schema(conn: sqlite3.Connection) -> None:
    score_cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(episode_scores)")
    }
    if "status" not in score_cols:
        conn.execute(
            "ALTER TABLE episode_scores ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'"
        )
    if "error" not in score_cols:
        conn.execute("ALTER TABLE episode_scores ADD COLUMN error TEXT")
    conn.commit()


def _db(path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a thread-local connection, creating it if necessary."""
    key = str(path.resolve())
    conns = getattr(_local, "conns", None)
    if conns is None:
        _local.conns = {}
        conns = _local.conns
    if key not in conns:
        conn = _connect(path)
        conns[key] = conn
        with _registry_lock:
            _all_conns[key] = conn
    return conns[key]


def close_all() -> None:
    """Close all known SQLite connections so server shutdown can exit cleanly."""
    with _registry_lock:
        items = list(_all_conns.items())
        _all_conns.clear()
    for key, conn in items:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    conns = getattr(_local, "conns", None)
    if isinstance(conns, dict):
        conns.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pack(value: object) -> str | None:
    """Serialise lists/dicts to JSON strings for storage."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value  # type: ignore[return-value]


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Deserialise JSON columns back to Python objects
    for col in ("cluster_span", "incident_span", "context_window",
                "clip_frame_indices", "extra"):
        raw = d.get(col)
        if raw is not None:
            try:
                d[col] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
    # Drop nulls and internal bookkeeping columns callers don't need
    d = {k: v for k, v in d.items() if v is not None and k != "created_at"}
    return d


# ── Public API ────────────────────────────────────────────────────────────────

def save_moment(moment: dict, path: Path = DB_PATH) -> None:
    """Upsert a moment.  Unknown fields are packed into the `extra` JSON blob."""
    known = {k: moment[k] for k in _KNOWN_COLUMNS if k in moment}
    extra_fields = {k: v for k, v in moment.items() if k not in _KNOWN_COLUMNS}

    db = _db(path)
    db.execute(
        """INSERT OR REPLACE INTO moments (
               episode_id, frame_index, cluster_id, cluster_size,
               label, verdict,
               cluster_span, incident_span, context_window,
               clip_pre_s, clip_post_s, clip_fps, clip_strategy,
               clip_frame_indices, clip_anchor_timestamp_s,
               clip_start_timestamp_s, clip_end_timestamp_s,
               extra
           ) VALUES (
               :episode_id, :frame_index, :cluster_id, :cluster_size,
               :label, :verdict,
               :cluster_span, :incident_span, :context_window,
               :clip_pre_s, :clip_post_s, :clip_fps, :clip_strategy,
               :clip_frame_indices, :clip_anchor_timestamp_s,
               :clip_start_timestamp_s, :clip_end_timestamp_s,
               :extra
           )""",
        {
            **{k: _pack(known.get(k)) for k in _KNOWN_COLUMNS},
            "extra": json.dumps(extra_fields) if extra_fields else None,
        },
    )
    db.commit()


def load_moments(path: Path = DB_PATH) -> Iterator[dict]:
    """Yield every moment ordered by (episode_id, frame_index)."""
    if not Path(path).exists():
        return
    db = _db(path)
    for row in db.execute(
        "SELECT * FROM moments ORDER BY episode_id, frame_index"
    ):
        yield _row_to_dict(row)


def get_moment(episode_id: int, frame_index: int,
               path: Path = DB_PATH) -> dict | None:
    """O(1) indexed lookup by primary key."""
    if not Path(path).exists():
        return None
    row = _db(path).execute(
        "SELECT * FROM moments WHERE episode_id = ? AND frame_index = ?",
        (episode_id, frame_index),
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_episode_moments(episode_id: int,
                        path: Path = DB_PATH) -> list[dict]:
    """Return all moments for an episode, sorted by frame_index."""
    if not Path(path).exists():
        return []
    rows = _db(path).execute(
        "SELECT * FROM moments WHERE episode_id = ? ORDER BY frame_index",
        (episode_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ── Resume support ────────────────────────────────────────────────────────────

def mark_episode_scanned(episode_id: int, frame_count: int,
                         flagged_count: int, path: Path = DB_PATH) -> None:
    """Record that an episode has been fully scored."""
    db = _db(path)
    db.execute(
        """INSERT OR REPLACE INTO scanned_episodes
               (episode_id, frame_count, flagged_count)
           VALUES (?, ?, ?)""",
        (episode_id, frame_count, flagged_count),
    )
    db.commit()


def get_scanned_episode_ids(path: Path = DB_PATH) -> set[int]:
    """Return the set of episode_ids that have already been fully scanned."""
    if not Path(path).exists():
        return set()
    rows = _db(path).execute(
        "SELECT episode_id FROM scanned_episodes"
    ).fetchall()
    return {row[0] for row in rows}


def get_scanned_episodes(path: Path = DB_PATH) -> list[dict]:
    """Return scanned episode summaries sorted by episode_id."""
    if not Path(path).exists():
        return []
    rows = _db(path).execute(
        """
        SELECT episode_id, frame_count, flagged_count, scanned_at
        FROM scanned_episodes
        ORDER BY episode_id
        """
    ).fetchall()
    return [
        {
            "episode_id": int(row["episode_id"]),
            "frame_count": int(row["frame_count"] or 0),
            "flagged_count": int(row["flagged_count"] or 0),
            "scanned_at": float(row["scanned_at"] or 0.0),
        }
        for row in rows
    ]


def save_episode_scores(
    episode_id: int,
    score_trace: list[dict],
    path: Path = DB_PATH,
) -> None:
    db = _db(path)
    db.execute(
        """
        INSERT OR REPLACE INTO episode_scores (episode_id, score_trace, status, error, updated_at)
        VALUES (?, ?, 'ready', NULL, unixepoch('now', 'subsec'))
        """,
        (int(episode_id), json.dumps(score_trace)),
    )
    db.commit()


def set_episode_score_state(
    episode_id: int,
    status: str,
    error: str | None = None,
    path: Path = DB_PATH,
) -> None:
    db = _db(path)
    row = db.execute(
        "SELECT score_trace FROM episode_scores WHERE episode_id = ?",
        (int(episode_id),),
    ).fetchone()
    score_trace = row["score_trace"] if row and row["score_trace"] is not None else "[]"
    db.execute(
        """
        INSERT OR REPLACE INTO episode_scores (episode_id, score_trace, status, error, updated_at)
        VALUES (?, ?, ?, ?, unixepoch('now', 'subsec'))
        """,
        (int(episode_id), score_trace, status, error),
    )
    db.commit()


def get_episode_score_record(episode_id: int, path: Path = DB_PATH) -> dict:
    if not Path(path).exists():
        return {"state": "pending", "trace": [], "error": None}
    row = _db(path).execute(
        "SELECT score_trace, status, error, updated_at FROM episode_scores WHERE episode_id = ?",
        (int(episode_id),),
    ).fetchone()
    if row is None:
        return {"state": "pending", "trace": [], "error": None}
    try:
        payload = json.loads(row["score_trace"] or "[]")
    except json.JSONDecodeError:
        payload = []
    trace = payload if isinstance(payload, list) else []
    state = str(row["status"] or "pending")
    # Normalize legacy or interrupted rows: a populated trace is a usable ready state.
    if trace and state in {"pending", "scoring"}:
        set_episode_score_state(int(episode_id), "ready", None, path)
        state = "ready"
    return {
        "state": state,
        "trace": trace,
        "error": row["error"],
        "updated_at": float(row["updated_at"] or 0.0),
    }


def get_episode_scores(episode_id: int, path: Path = DB_PATH) -> list[dict]:
    return list(get_episode_score_record(episode_id, path).get("trace", []))


def save_episode_media(
    episode_id: int,
    video_url: str | None,
    camera_videos: dict[str, str] | None,
    path: Path = DB_PATH,
) -> None:
    db = _db(path)
    db.execute(
        """
        INSERT OR REPLACE INTO episode_media (episode_id, video_url, camera_videos, updated_at)
        VALUES (?, ?, ?, unixepoch('now', 'subsec'))
        """,
        (
            int(episode_id),
            video_url,
            json.dumps(camera_videos or {}),
        ),
    )
    db.commit()


def get_episode_media(episode_id: int, path: Path = DB_PATH) -> dict:
    if not Path(path).exists():
        return {}
    row = _db(path).execute(
        "SELECT video_url, camera_videos FROM episode_media WHERE episode_id = ?",
        (int(episode_id),),
    ).fetchone()
    if row is None:
        return {}
    try:
        camera_videos = json.loads(row["camera_videos"] or "{}")
    except json.JSONDecodeError:
        camera_videos = {}
    return {
        "video_url": row["video_url"],
        "camera_videos": camera_videos if isinstance(camera_videos, dict) else {},
    }


def clear_runtime_tables(path: Path = DB_PATH) -> None:
    db = _db(path)
    db.execute("DELETE FROM episode_scores")
    db.execute("DELETE FROM episode_media")
    db.commit()
