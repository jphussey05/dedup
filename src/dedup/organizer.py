"""Move surviving images into a consolidated YYYY/YYYY-MM-DD_HHMMSS.ext archive.

Runs AFTER `review --execute` has removed duplicates. Every row still in the DB
is a keeper; this module renames and relocates those files on disk and updates
their stored paths accordingly.

Date source priority (per plan swift-hopping-owl.md):
    1. EXIF ``DateTimeOriginal`` / ``DateTimeDigitized``
    2. ``file_mtime`` (scanner-captured filesystem mtime)
    3. skipped with a warning if neither parses
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from dedup.db import Database
from dedup.pathutil import normalize_path


@dataclass
class MoveDecision:
    image_id: int
    src_path: str        # as stored in DB (forward-slash, lowercased on Windows)
    dest_path: str       # absolute, OS-native (backslash UNC on Windows)
    date_source: str     # "exif" | "mtime"
    dest_year: str
    file_size: int


# --- Date parsing ---


def _parse_date(
    exif_date: str | None, file_mtime: str | None
) -> tuple[datetime | None, str]:
    """Return (datetime, source). EXIF first, then ISO mtime, else (None, 'none')."""
    if exif_date:
        try:
            return datetime.strptime(exif_date, "%Y:%m:%d %H:%M:%S"), "exif"
        except ValueError:
            # Malformed EXIF (e.g. "0000:00:00 00:00:00") — fall through
            pass
    if file_mtime:
        try:
            return datetime.fromisoformat(file_mtime), "mtime"
        except ValueError:
            pass
    return None, "none"


# --- Destination naming ---


def _natural_dest(dt: datetime, ext: str, dest_root: Path) -> Path:
    """The unsuffixed target for (dt, ext) — the canonical name before any collision handling."""
    ext = ext.lower()
    if not ext.startswith("."):
        ext = "." + ext
    stem = dt.strftime("%Y-%m-%d_%H%M%S")
    return dest_root / f"{dt.year:04d}" / f"{stem}{ext}"


def _build_dest(
    dt: datetime,
    ext: str,
    dest_root: Path,
    taken: set[Path],
) -> Path:
    """Compose ``dest_root/YYYY/YYYY-MM-DD_HHMMSS.ext``, suffixing on collision.

    `taken` tracks destinations already claimed in this planning pass so two
    sources that share a timestamp don't target the same file. Existing files
    on disk are also treated as occupied (protects against clobbering unrelated
    content left behind by a prior run).
    """
    candidate = _natural_dest(dt, ext, dest_root)
    stem_base = candidate.stem
    ext_norm = candidate.suffix
    year_dir = candidate.parent

    suffix = 0
    while candidate in taken or candidate.exists():
        suffix += 1
        candidate = year_dir / f"{stem_base}_{suffix:03d}{ext_norm}"
    return candidate


def _denormalize(db_path: str) -> str:
    """Convert the DB's forward-slash path into an OS-native path.

    On Windows this turns ``//server/share/x.jpg`` into ``\\\\server\\share\\x.jpg``,
    which is the form the Win32 file APIs (including ``shutil.move``) actually accept.
    On POSIX the result is equivalent to the input.
    """
    return str(Path(db_path).resolve())


# --- Planning ---


def _already_at_dest(src_db_path: str, dest_native: Path) -> bool:
    """True when a row's current path already matches the target (idempotency)."""
    return normalize_path(str(dest_native)) == src_db_path


def plan_moves(
    db: Database,
    dest_root: Path,
    limit: int | None = None,
) -> tuple[list[MoveDecision], int]:
    """Build a MoveDecision for every surviving image that has a usable date.

    Returns (decisions, skipped_no_date).
    """
    rows = db.get_surviving_images()
    if limit is not None:
        rows = rows[:limit]

    decisions: list[MoveDecision] = []
    taken: set[Path] = set()
    skipped_no_date = 0

    for row in rows:
        dt, source = _parse_date(row["exif_date"], row["file_mtime"])
        if dt is None:
            skipped_no_date += 1
            continue

        ext = Path(row["filename"] or row["path"]).suffix or ".jpg"

        # Idempotency check: compare to the UNSUFFIXED destination. If the row's
        # current path already matches the canonical name, there's nothing to do —
        # without this guard, _build_dest would see the file on disk (our own!)
        # and falsely bump to _001.
        if _already_at_dest(row["path"], _natural_dest(dt, ext, dest_root)):
            continue

        dest = _build_dest(dt, ext, dest_root, taken)
        taken.add(dest)
        decisions.append(
            MoveDecision(
                image_id=row["id"],
                src_path=row["path"],
                dest_path=str(dest),
                date_source=source,
                dest_year=f"{dt.year:04d}",
                file_size=row["file_size"] or 0,
            )
        )

    return decisions, skipped_no_date


# --- Preview & export ---


def _format_size(size: int | None) -> str:
    if not size:
        return "0 B"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024 / 1024 / 1024:.2f} GB"


def preview_moves(
    decisions: list[MoveDecision],
    console: Console,
    skipped_no_date: int = 0,
    max_rows: int = 20,
) -> None:
    """Print a compact Rich table showing the first N planned moves."""
    if not decisions:
        console.print("[dim]Nothing to move.[/dim]")
        if skipped_no_date:
            console.print(
                f"[yellow]{skipped_no_date} images skipped (no EXIF or mtime).[/yellow]"
            )
        return

    total_bytes = sum(d.file_size for d in decisions)

    table = Table(title=f"Move preview (first {min(max_rows, len(decisions))} of {len(decisions)})")
    table.add_column("Year", no_wrap=True)
    table.add_column("Source", overflow="fold")
    table.add_column("→", no_wrap=True)
    table.add_column("Destination", overflow="fold")
    table.add_column("Date", no_wrap=True)

    for d in decisions[:max_rows]:
        table.add_row(
            d.dest_year,
            d.src_path,
            "→",
            d.dest_path,
            d.date_source,
        )

    console.print(table)
    console.print(
        f"[bold]Total:[/bold] {len(decisions)} files to move, "
        f"~{_format_size(total_bytes)}"
    )
    if skipped_no_date:
        console.print(
            f"[yellow]{skipped_no_date} images skipped (no EXIF or mtime) — "
            "they stay put.[/yellow]"
        )
    if len(decisions) > max_rows:
        console.print(f"[dim]({len(decisions) - max_rows} more not shown)[/dim]")


def export_plan(decisions: list[MoveDecision], path: Path) -> None:
    """Dump the plan to CSV for offline inspection."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "src_path", "dest_path",
            "date_source", "dest_year", "file_size",
        ])
        for d in decisions:
            writer.writerow([
                d.image_id, d.src_path, d.dest_path,
                d.date_source, d.dest_year, d.file_size,
            ])


# --- Execution ---


def execute_moves(
    db: Database,
    decisions: list[MoveDecision],
    console: Console,
) -> tuple[int, int]:
    """Actually move files and update DB paths.

    Returns (moved_count, error_count).
    """
    moved = 0
    errors = 0
    batch_since_commit = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Moving", total=len(decisions))
        for d in decisions:
            src_native = _denormalize(d.src_path)
            dest_native = d.dest_path
            try:
                Path(dest_native).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src_native, dest_native)

                # Same defensive check we use in the deleter: shell APIs can
                # "succeed" on mangled paths without doing anything.
                if not Path(dest_native).exists():
                    raise OSError(f"move reported OK but destination missing: {dest_native}")
                if Path(src_native).exists():
                    raise OSError(f"move reported OK but source still present: {src_native}")

                db.update_path(d.image_id, normalize_path(dest_native))
                moved += 1
                batch_since_commit += 1
                if batch_since_commit >= 500:
                    db.conn.commit()
                    batch_since_commit = 0
            except Exception as exc:
                errors += 1
                console.print(f"[red]Failed:[/red] {d.src_path} → {d.dest_path} ({exc})")
            progress.advance(task)

    db.conn.commit()
    db.checkpoint()
    return moved, errors
