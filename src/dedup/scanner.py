"""Stage 0: File discovery and EXIF extraction."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from dedup.db import Database
from dedup.pathutil import normalize_path


def _extract_exif_date(img: Image.Image) -> str | None:
    """Extract the original date from EXIF data."""
    try:
        exif = img.getexif()
        if exif:
            # 36867 = DateTimeOriginal, 36868 = DateTimeDigitized
            for tag in (36867, 36868):
                val = exif.get(tag)
                if val and isinstance(val, str):
                    return val
    except Exception:
        pass
    return None


def _get_image_info(filepath: str) -> tuple[int | None, int | None, str | None, str | None]:
    """Get image dimensions, format, and EXIF date without loading pixel data."""
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            fmt = img.format
            exif_date = _extract_exif_date(img)
            return width, height, fmt, exif_date
    except Exception:
        return None, None, None, None


def _walk_directory(root: str, extensions: set[str], recursive: bool):
    """Walk directory using os.scandir for performance on network shares."""
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in extensions:
                            yield entry
                    elif recursive and entry.is_dir(follow_symlinks=False):
                        yield from _walk_directory(entry.path, extensions, recursive)
                except (PermissionError, OSError):
                    continue
    except (PermissionError, OSError):
        return


def scan_directory(
    db: Database,
    root: str,
    extensions: set[str],
    recursive: bool = True,
) -> None:
    """Scan a directory for images and insert them into the database."""
    from rich.console import Console

    console = Console()
    root_path = str(Path(root).resolve())
    norm_root = normalize_path(root)

    session_id = db.start_scan_session(norm_root)

    # First pass: count files for progress bar
    console.print(f"[bold]Scanning:[/bold] {root_path}")
    console.print("[dim]Counting files...[/dim]")
    all_entries = list(_walk_directory(root_path, extensions, recursive))
    files_found = len(all_entries)
    console.print(f"[dim]Found {files_found} image files[/dim]")

    files_new = 0
    files_skipped = 0
    batch_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Indexing images", total=files_found)

        for entry in all_entries:
            try:
                norm_path = normalize_path(entry.path)

                if db.image_exists(norm_path):
                    files_skipped += 1
                    progress.advance(task)
                    continue

                stat = entry.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

                width, height, fmt, exif_date = _get_image_info(entry.path)

                db.insert_image(
                    path=norm_path,
                    filename=entry.name,
                    file_size=stat.st_size,
                    width=width,
                    height=height,
                    fmt=fmt,
                    exif_date=exif_date,
                    file_mtime=mtime,
                )
                files_new += 1
                batch_count += 1

                if batch_count >= 500:
                    db.conn.commit()
                    batch_count = 0

            except Exception as e:
                console.print(f"[yellow]Warning: {entry.path}: {e}[/yellow]")

            progress.advance(task)

    db.conn.commit()
    db.finish_scan_session(session_id, files_found, files_new, files_skipped)
    console.print(
        f"[green]Done![/green] {files_new} new, {files_skipped} skipped"
    )
