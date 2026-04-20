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

    `taken` is authoritative: it should already contain every path that exists
    on disk under ``dest_root`` (seeded by ``_snapshot_dest``) plus every
    destination reserved earlier in this planning pass. We deliberately do NOT
    call ``candidate.exists()`` here — for large UNC runs that's one SMB stat
    per image, which dwarfs the actual work.
    """
    candidate = _natural_dest(dt, ext, dest_root)
    stem_base = candidate.stem
    ext_norm = candidate.suffix
    year_dir = candidate.parent

    suffix = 0
    while candidate in taken:
        suffix += 1
        candidate = year_dir / f"{stem_base}_{suffix:03d}{ext_norm}"
    return candidate


def _snapshot_dest(dest_root: Path) -> set[Path]:
    """One-shot listing of every file already under ``dest_root/YYYY/``.

    Planning previously called ``Path(candidate).exists()`` per image — fine
    locally, catastrophic over SMB (one network round-trip per survivor). This
    turns N stats into ~K directory listings where K = number of year folders,
    which on a fresh archive means one ``iterdir`` on ``dest_root`` itself.
    """
    occupied: set[Path] = set()
    try:
        if not dest_root.exists():
            return occupied
        for year_dir in dest_root.iterdir():
            if not year_dir.is_dir():
                continue
            try:
                for f in year_dir.iterdir():
                    if f.is_file():
                        occupied.add(f)
            except OSError:
                continue
    except OSError:
        # Unreachable dest (offline NAS, permission error) — treat as empty.
        # The subsequent mkdir during execute will raise the real error.
        pass
    return occupied


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


def _normalize_exclude_prefixes(exclude_sources: list[str] | None) -> list[str]:
    """Turn user-supplied exclude paths into DB-canonical prefix strings.

    The DB stores paths via ``normalize_path`` (forward slash, lowercased on
    Windows). We match the same way, and append a trailing ``/`` so the prefix
    test respects directory boundaries — ``a/b`` excludes ``a/b/x`` but not
    ``a/bcd``.
    """
    if not exclude_sources:
        return []
    out: list[str] = []
    for s in exclude_sources:
        norm = normalize_path(s).rstrip("/")
        if norm:
            out.append(norm + "/")
    return out


def plan_moves(
    db: Database,
    dest_root: Path,
    limit: int | None = None,
    console: Console | None = None,
    exclude_sources: list[str] | None = None,
) -> tuple[list[MoveDecision], int]:
    """Build a MoveDecision for every surviving image that has a usable date.

    Returns (decisions, skipped_no_date).

    ``exclude_sources`` is a list of path prefixes; any row whose current path
    starts with one of them is left alone. This is the blacklist equivalent of
    choosing scan scope after the fact — useful for folders you never want
    reorganized (backups, external-drive mirrors, non-photo archives).

    For large DBs (100k+ rows) the planning phase was previously silent and
    looked like a hang — this now snapshots the destination once, then streams
    the row loop through a visible progress bar.
    """
    if console is None:
        console = Console()

    excludes = _normalize_exclude_prefixes(exclude_sources)

    with console.status("[bold]Loading surviving images from DB[/bold]"):
        rows = db.get_surviving_images()

    if excludes:
        before = len(rows)
        rows = [r for r in rows if not any(r["path"].startswith(e) for e in excludes)]
        skipped_excluded = before - len(rows)
        if skipped_excluded:
            console.print(
                f"[dim]Excluded {skipped_excluded:,} row(s) under "
                f"{len(excludes)} blacklisted prefix(es).[/dim]"
            )

    if limit is not None:
        rows = rows[:limit]

    if not rows:
        return [], 0

    with console.status(f"[bold]Snapshotting {dest_root}[/bold] (one-shot listing)"):
        taken = _snapshot_dest(dest_root)

    decisions: list[MoveDecision] = []
    skipped_no_date = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Planning", total=len(rows))
        for row in rows:
            dt, source = _parse_date(row["exif_date"], row["file_mtime"])
            if dt is None:
                skipped_no_date += 1
                progress.advance(task)
                continue

            ext = Path(row["filename"] or row["path"]).suffix or ".jpg"

            # Idempotency check: compare to the UNSUFFIXED destination. If the
            # row's current path already matches the canonical name, nothing to
            # do — without this guard, _build_dest would see our own file in
            # the occupied set and falsely bump to _001.
            natural = _natural_dest(dt, ext, dest_root)
            if _already_at_dest(row["path"], natural):
                progress.advance(task)
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
            progress.advance(task)

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

    Ctrl+C is handled cleanly: the pending DB batch is committed in a
    ``finally`` so path updates for already-moved files aren't rolled back.
    Without this guard, up to ~499 files could end up moved on disk but
    still pointing at their old path in the DB — a subtle and expensive
    inconsistency to clean up later.
    """
    moved = 0
    errors = 0
    batch_since_commit = 0

    try:
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
    except KeyboardInterrupt:
        console.print(
            f"\n[yellow]Interrupted at {moved}/{len(decisions)} — "
            "committing pending DB updates before exit.[/yellow]"
        )
        raise
    finally:
        # Always commit — interrupts and exceptions should not roll back the
        # path updates for files already moved on disk.
        db.conn.commit()
        db.checkpoint()

    return moved, errors
