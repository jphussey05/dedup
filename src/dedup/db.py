"""SQLite database management for dedup."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    filename    TEXT NOT NULL,
    file_size   INTEGER NOT NULL,
    width       INTEGER,
    height      INTEGER,
    format      TEXT,
    exif_date   TEXT,
    sha256      TEXT,
    phash       TEXT,
    dhash       TEXT,
    phash_int   INTEGER,
    dhash_int   INTEGER,
    embedding   BLOB,
    scanned_at  TEXT NOT NULL DEFAULT (datetime('now')),
    hashed_at   TEXT,
    phashed_at  TEXT,
    embedded_at TEXT,
    file_mtime  TEXT,
    error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256) WHERE sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_images_phash_int ON images(phash_int) WHERE phash_int IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_images_dhash_int ON images(dhash_int) WHERE dhash_int IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_images_file_size ON images(file_size);

CREATE TABLE IF NOT EXISTS duplicate_groups (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    match_type  TEXT NOT NULL CHECK (match_type IN ('exact', 'perceptual', 'cnn')),
    confidence  REAL NOT NULL DEFAULT 1.0,
    image_count INTEGER NOT NULL DEFAULT 2,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dg_match_type ON duplicate_groups(match_type);

CREATE TABLE IF NOT EXISTS duplicate_pairs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id    INTEGER NOT NULL REFERENCES duplicate_groups(id) ON DELETE CASCADE,
    image_id_1  INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    image_id_2  INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    similarity  REAL NOT NULL DEFAULT 1.0,
    UNIQUE(group_id, image_id_1, image_id_2)
);

CREATE INDEX IF NOT EXISTS idx_dp_group ON duplicate_pairs(group_id);
CREATE INDEX IF NOT EXISTS idx_dp_image1 ON duplicate_pairs(image_id_1);
CREATE INDEX IF NOT EXISTS idx_dp_image2 ON duplicate_pairs(image_id_2);

CREATE TABLE IF NOT EXISTS scan_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    root_path   TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    files_found INTEGER DEFAULT 0,
    files_new   INTEGER DEFAULT 0,
    files_skipped INTEGER DEFAULT 0
);
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def init_schema(self) -> None:
        """Create all tables and indexes."""
        self.conn.executescript(SCHEMA_SQL)
        # Record schema version
        existing = self.conn.execute(
            "SELECT version FROM schema_version WHERE version = ?", (SCHEMA_VERSION,)
        ).fetchone()
        if not existing:
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
            self.conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for a database transaction."""
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # --- Image operations ---

    def image_exists(self, path: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM images WHERE path = ?", (path,)
        ).fetchone()
        return row is not None

    def insert_image(
        self,
        path: str,
        filename: str,
        file_size: int,
        width: int | None = None,
        height: int | None = None,
        fmt: str | None = None,
        exif_date: str | None = None,
        file_mtime: str | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO images
               (path, filename, file_size, width, height, format, exif_date, file_mtime)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (path, filename, file_size, width, height, fmt, exif_date, file_mtime),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_unhashed_images(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, path, file_size FROM images WHERE sha256 IS NULL AND error IS NULL"
        ).fetchall()

    def get_duplicate_size_images(self) -> list[sqlite3.Row]:
        """Get images that share a file_size with at least one other image (unhashed only)."""
        return self.conn.execute(
            """SELECT i.id, i.path, i.file_size FROM images i
               WHERE i.sha256 IS NULL AND i.error IS NULL
               AND i.file_size IN (
                   SELECT file_size FROM images
                   WHERE sha256 IS NULL AND error IS NULL
                   GROUP BY file_size HAVING COUNT(*) > 1
               )"""
        ).fetchall()

    def get_unique_size_images(self) -> list[sqlite3.Row]:
        """Get images with unique file sizes (no possible exact duplicate)."""
        return self.conn.execute(
            """SELECT i.id, i.path FROM images i
               WHERE i.sha256 IS NULL AND i.error IS NULL
               AND i.file_size IN (
                   SELECT file_size FROM images
                   WHERE sha256 IS NULL AND error IS NULL
                   GROUP BY file_size HAVING COUNT(*) = 1
               )"""
        ).fetchall()

    def update_sha256(self, image_id: int, sha256: str) -> None:
        self.conn.execute(
            "UPDATE images SET sha256 = ?, hashed_at = datetime('now') WHERE id = ?",
            (sha256, image_id),
        )

    def mark_unique_size_hashed(self, image_id: int) -> None:
        """Mark an image with unique file size as hashed (use path as pseudo-hash)."""
        self.conn.execute(
            "UPDATE images SET sha256 = 'unique_size', hashed_at = datetime('now') WHERE id = ?",
            (image_id,),
        )

    def update_error(self, image_id: int, error: str) -> None:
        self.conn.execute(
            "UPDATE images SET error = ? WHERE id = ?", (error, image_id)
        )

    def get_unphashed_images(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, path FROM images WHERE phash IS NULL AND error IS NULL"
        ).fetchall()

    def update_phash(
        self,
        image_id: int,
        phash: str,
        phash_int: int,
        dhash: str,
        dhash_int: int,
    ) -> None:
        self.conn.execute(
            """UPDATE images SET phash = ?, phash_int = ?, dhash = ?, dhash_int = ?,
               phashed_at = datetime('now') WHERE id = ?""",
            (phash, phash_int, dhash, dhash_int, image_id),
        )

    def get_all_phashes(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, phash_int, dhash_int FROM images WHERE phash_int IS NOT NULL"
        ).fetchall()

    def get_unembedded_images(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, path FROM images WHERE embedding IS NULL AND error IS NULL"
        ).fetchall()

    def update_embedding(self, image_id: int, embedding_blob: bytes) -> None:
        self.conn.execute(
            "UPDATE images SET embedding = ?, embedded_at = datetime('now') WHERE id = ?",
            (embedding_blob, image_id),
        )

    def get_all_embeddings(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, embedding FROM images WHERE embedding IS NOT NULL"
        ).fetchall()

    # --- Duplicate group operations ---

    def clear_groups(self, match_type: str) -> None:
        """Remove all existing groups of a given type before re-computing."""
        group_ids = self.conn.execute(
            "SELECT id FROM duplicate_groups WHERE match_type = ?", (match_type,)
        ).fetchall()
        for row in group_ids:
            self.conn.execute("DELETE FROM duplicate_pairs WHERE group_id = ?", (row["id"],))
        self.conn.execute("DELETE FROM duplicate_groups WHERE match_type = ?", (match_type,))

    def insert_duplicate_group(
        self, match_type: str, confidence: float, image_count: int
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO duplicate_groups (match_type, confidence, image_count)
               VALUES (?, ?, ?)""",
            (match_type, confidence, image_count),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def insert_duplicate_pair(
        self, group_id: int, image_id_1: int, image_id_2: int, similarity: float
    ) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO duplicate_pairs (group_id, image_id_1, image_id_2, similarity)
               VALUES (?, ?, ?, ?)""",
            (group_id, image_id_1, image_id_2, similarity),
        )

    def get_exact_duplicates(self) -> list[sqlite3.Row]:
        """Get SHA256 values that appear more than once."""
        return self.conn.execute(
            """SELECT sha256, COUNT(*) as cnt FROM images
               WHERE sha256 IS NOT NULL AND sha256 != 'unique_size'
               GROUP BY sha256 HAVING COUNT(*) > 1"""
        ).fetchall()

    def get_images_by_hash(self, sha256: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, path, file_size, filename FROM images WHERE sha256 = ?", (sha256,)
        ).fetchall()

    # --- Reporting queries ---

    def get_duplicate_report(
        self, match_type: str | None = None, min_confidence: float = 0.0
    ) -> list[dict[str, Any]]:
        """Get duplicate groups with their image details for reporting."""
        where_clauses = ["dg.confidence >= ?"]
        params: list[Any] = [min_confidence]

        if match_type and match_type != "all":
            where_clauses.append("dg.match_type = ?")
            params.append(match_type)

        where_sql = " AND ".join(where_clauses)

        rows = self.conn.execute(
            f"""SELECT dg.id as group_id, dg.match_type, dg.confidence, dg.image_count,
                       dp.image_id_1, dp.image_id_2, dp.similarity,
                       i1.path as path1, i1.file_size as size1, i1.filename as name1,
                       i2.path as path2, i2.file_size as size2, i2.filename as name2
                FROM duplicate_groups dg
                JOIN duplicate_pairs dp ON dp.group_id = dg.id
                JOIN images i1 ON i1.id = dp.image_id_1
                JOIN images i2 ON i2.id = dp.image_id_2
                WHERE {where_sql}
                ORDER BY dg.match_type, dg.id""",
            params,
        ).fetchall()

        return [dict(r) for r in rows]

    def get_status(self) -> dict[str, Any]:
        """Get current processing status counts."""
        total = self.conn.execute("SELECT COUNT(*) as n FROM images").fetchone()["n"]
        hashed = self.conn.execute(
            "SELECT COUNT(*) as n FROM images WHERE sha256 IS NOT NULL"
        ).fetchone()["n"]
        phashed = self.conn.execute(
            "SELECT COUNT(*) as n FROM images WHERE phash IS NOT NULL"
        ).fetchone()["n"]
        embedded = self.conn.execute(
            "SELECT COUNT(*) as n FROM images WHERE embedding IS NOT NULL"
        ).fetchone()["n"]
        errored = self.conn.execute(
            "SELECT COUNT(*) as n FROM images WHERE error IS NOT NULL"
        ).fetchone()["n"]

        exact_groups = self.conn.execute(
            "SELECT COUNT(*) as n FROM duplicate_groups WHERE match_type = 'exact'"
        ).fetchone()["n"]
        perceptual_groups = self.conn.execute(
            "SELECT COUNT(*) as n FROM duplicate_groups WHERE match_type = 'perceptual'"
        ).fetchone()["n"]
        cnn_groups = self.conn.execute(
            "SELECT COUNT(*) as n FROM duplicate_groups WHERE match_type = 'cnn'"
        ).fetchone()["n"]

        return {
            "total_images": total,
            "hashed": hashed,
            "phashed": phashed,
            "embedded": embedded,
            "errored": errored,
            "exact_groups": exact_groups,
            "perceptual_groups": perceptual_groups,
            "cnn_groups": cnn_groups,
        }

    # --- Scan session operations ---

    def start_scan_session(self, root_path: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO scan_sessions (root_path) VALUES (?)", (root_path,)
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def finish_scan_session(
        self, session_id: int, files_found: int, files_new: int, files_skipped: int
    ) -> None:
        self.conn.execute(
            """UPDATE scan_sessions
               SET finished_at = datetime('now'), files_found = ?, files_new = ?, files_skipped = ?
               WHERE id = ?""",
            (files_found, files_new, files_skipped, session_id),
        )
        self.conn.commit()
