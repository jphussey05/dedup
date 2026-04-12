"""Stage 0: File discovery and EXIF extraction."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from rich.live import Live

# Raise Pillow's decompression bomb limit for trusted local/NAS files.
# Default is ~89MP which is easily exceeded by large TIFFs and scanned photos.
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500 megapixels
from rich.table import Table

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


def _get_image_info(
    filepath: str,
) -> tuple[int | None, int | None, str | None, str | None]:
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
                        yield from _walk_directory(
                            entry.path, extensions, recursive
                        )
                except (PermissionError, OSError):
                    continue
    except (PermissionError, OSError):
        return


def _make_status_table(
    found: int, new: int, skipped: int, errors: int, current_dir: str
) -> Table:
    """Build a live status table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Images found", str(found))
    table.add_row("New", f"[green]{new}[/green]")
    table.add_row("Skipped (already in DB)", str(skipped))
    if errors:
        table.add_row("Errors", f"[yellow]{errors}[/yellow]")
    table.add_row("Current folder", f"[dim]{current_dir}[/dim]")
    return table


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

    console.print(f"[bold]Scanning:[/bold] {root_path}")

    files_found = 0
    files_new = 0
    files_skipped = 0
    files_errors = 0
    batch_count = 0
    current_dir = root_path

    with Live(
        _make_status_table(0, 0, 0, 0, root_path),
        console=console,
        refresh_per_second=4,
    ) as live:
        for entry in _walk_directory(root_path, extensions, recursive):
            files_found += 1

            # Update current directory display
            entry_dir = os.path.dirname(entry.path)
            if entry_dir != current_dir:
                current_dir = entry_dir

            try:
                norm_path = normalize_path(entry.path)

                if db.image_exists(norm_path):
                    files_skipped += 1
                    if files_found % 50 == 0:
                        live.update(
                            _make_status_table(
                                files_found, files_new, files_skipped,
                                files_errors, current_dir,
                            )
                        )
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

            except Exception:
                files_errors += 1

            if files_found % 50 == 0:
                live.update(
                    _make_status_table(
                        files_found, files_new, files_skipped,
                        files_errors, current_dir,
                    )
                )

    db.conn.commit()
    db.finish_scan_session(
        session_id, files_found, files_new, files_skipped
    )
    console.print(
        f"[green]Done![/green] {files_found} found, "
        f"{files_new} new, {files_skipped} skipped"
    )
    if files_errors:
        console.print(f"[yellow]{files_errors} errors[/yellow]")
