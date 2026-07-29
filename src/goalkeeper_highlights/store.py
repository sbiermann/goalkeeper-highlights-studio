from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import Candidate


class AnalysisStore:
    """SQLite is the canonical analysis format.

    Raw detections, processed frames, candidates, profiling samples and pipeline
    state are stored independently. Expensive detection can therefore be reused.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE IF NOT EXISTS state (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS videos (
              id INTEGER PRIMARY KEY CHECK(id=1),
              path TEXT NOT NULL,
              duration REAL NOT NULL,
              fps REAL,
              width INTEGER,
              height INTEGER,
              frame_count INTEGER,
              decoder TEXT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS frames (
              frame_index INTEGER PRIMARY KEY,
              timestamp REAL NOT NULL,
              processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              detection_count INTEGER NOT NULL DEFAULT 0,
              person_count INTEGER NOT NULL DEFAULT 0,
              ball_count INTEGER NOT NULL DEFAULT 0,
              keeper_track_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS candidates (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS detections (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              frame_index INTEGER NOT NULL,
              timestamp REAL NOT NULL,
              track_id INTEGER,
              class_id INTEGER NOT NULL,
              confidence REAL NOT NULL,
              x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id);
            CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_index);
            CREATE TABLE IF NOT EXISTS profiling_samples (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              wall_seconds REAL NOT NULL,
              video_seconds REAL NOT NULL,
              frame_index INTEGER NOT NULL,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS benchmarks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              name TEXT NOT NULL,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              progress REAL NOT NULL DEFAULT 0,
              message TEXT NOT NULL DEFAULT '',
              payload TEXT NOT NULL DEFAULT '{}',
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def set_state(self, key: str, value: object) -> None:
        self.connection.execute(
            "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.connection.commit()

    def get_state(self, key: str, default: object = None) -> object:
        row = self.connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def save_video(self, *, path: Path, duration: float, fps: float, width: int, height: int, frame_count: int, decoder: str) -> None:
        self.connection.execute(
            """INSERT INTO videos(id,path,duration,fps,width,height,frame_count,decoder)
               VALUES(1,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET path=excluded.path,duration=excluded.duration,
               fps=excluded.fps,width=excluded.width,height=excluded.height,
               frame_count=excluded.frame_count,decoder=excluded.decoder""",
            (str(path), duration, fps, width, height, frame_count, decoder),
        )
        self.connection.commit()

    def replace_candidates(self, candidates: list[Candidate]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM candidates")
            self.connection.executemany(
                "INSERT INTO candidates(payload) VALUES(?)",
                [(json.dumps(c.as_dict(), ensure_ascii=False),) for c in candidates],
            )

    def load_candidates(self) -> list[Candidate]:
        rows = self.connection.execute("SELECT payload FROM candidates ORDER BY id").fetchall()
        return [Candidate(**json.loads(row[0])) for row in rows]

    def append_detections(self, rows: Iterable[tuple]) -> None:
        self.connection.executemany(
            "INSERT INTO detections(frame_index,timestamp,track_id,class_id,confidence,x1,y1,x2,y2) VALUES(?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.connection.commit()

    def append_frames(self, rows: Iterable[tuple]) -> None:
        self.connection.executemany(
            """INSERT INTO frames(frame_index,timestamp,detection_count,person_count,ball_count,keeper_track_id)
               VALUES(?,?,?,?,?,?) ON CONFLICT(frame_index) DO UPDATE SET
               timestamp=excluded.timestamp,detection_count=excluded.detection_count,
               person_count=excluded.person_count,ball_count=excluded.ball_count,
               keeper_track_id=excluded.keeper_track_id""",
            rows,
        )
        self.connection.commit()

    def append_profile_sample(self, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO profiling_samples(wall_seconds,video_seconds,frame_index,payload) VALUES(?,?,?,?)",
            (payload["wall_seconds"], payload["video_seconds"], payload["frame_index"], json.dumps(payload, ensure_ascii=False)),
        )
        self.connection.commit()

    def save_benchmark(self, name: str, payload: dict[str, Any]) -> None:
        self.connection.execute("INSERT INTO benchmarks(name,payload) VALUES(?,?)", (name, json.dumps(payload, ensure_ascii=False)))
        self.connection.commit()


    def recovery_observations(self) -> list[dict[str, Any]]:
        """Return per-frame keeper/ball geometry for the generic missed-action pass."""
        rows = self.connection.execute(
            """
            SELECT f.frame_index, f.timestamp, f.keeper_track_id,
                   k.x1 AS kx1, k.y1 AS ky1, k.x2 AS kx2, k.y2 AS ky2,
                   b.confidence AS ball_confidence, b.x1 AS bx1, b.y1 AS by1, b.x2 AS bx2, b.y2 AS by2
            FROM frames f
            JOIN detections k ON k.frame_index=f.frame_index AND k.class_id=0 AND k.track_id=f.keeper_track_id
            JOIN detections b ON b.frame_index=f.frame_index AND b.class_id=32
            ORDER BY f.timestamp, b.confidence DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def keeper_identity_timeline(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT frame_index,timestamp,keeper_track_id,person_count,ball_count FROM frames ORDER BY timestamp"
        ).fetchall()
        return [dict(row) for row in rows]

    def ball_detection_gaps(self, minimum_gap_seconds: float = 0.5) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT timestamp,ball_count FROM frames ORDER BY timestamp"
        ).fetchall()
        gaps: list[dict[str, Any]] = []
        start = None
        last = None
        for row in rows:
            ts = float(row["timestamp"])
            if int(row["ball_count"]) == 0:
                if start is None:
                    start = ts
                last = ts
            elif start is not None and last is not None:
                if last - start >= minimum_gap_seconds:
                    gaps.append({"start": start, "end": last, "duration": last-start})
                start = last = None
        if start is not None and last is not None and last-start >= minimum_gap_seconds:
            gaps.append({"start": start, "end": last, "duration": last-start})
        return gaps

    def checkpoint(self) -> None:
        """Flush the WAL so the database can be copied into a debug package."""
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(FULL)").fetchall()

    def upsert_job(self, job_id: str, status: str, progress: float, message: str, payload: dict | None = None) -> None:
        self.connection.execute(
            """
            INSERT INTO jobs(id,status,progress,message,payload) VALUES(?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              status=excluded.status, progress=excluded.progress, message=excluded.message,
              payload=excluded.payload, updated_at=CURRENT_TIMESTAMP
            """,
            (job_id, status, progress, message, json.dumps(payload or {}, ensure_ascii=False)),
        )
        self.connection.commit()
