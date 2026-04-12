"""Stage 1: SHA256 exact hash computation."""

from __future__ import annotations

import hashlib

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from dedup.db import Database
from dedup.pathutil import get_read_chunk_size


def _compute_sha256(path: str) -> str:
    """Compute SHA256 hash of a file using streaming reads."""
    chunk_size = get_read_chunk_size(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_images(db: Database, batch_size: int = 1000) -> None:
    """Compute SHA256 hashes for all unhashed images.

    Optimization: only hash files that share a file_size with another file,
    since unique-sized files cannot be exact duplicates.
    """
    from rich.console import Console

    console = Console()

    # Mark unique-size files as "hashed" (they can't have exact duplicates)
    unique_size_images = db.get_unique_size_images()
    if unique_size_images:
        console.print(
            f"[dim]Skipping {len(unique_size_images)} images with unique file sizes[/dim]"
        )
        for row in unique_size_images:
            db.mark_unique_size_hashed(row["id"])
        db.conn.commit()

    # Hash only files that share a size with at least one other file
    candidates = db.get_duplicate_size_images()
    if not candidates:
        console.print("[green]All images already hashed.[/green]")
        return

    console.print(f"[bold]Hashing {len(candidates)} images[/bold] (shared file sizes only)")

    batch_count = 0
    errors = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Computing SHA256", total=len(candidates))

        for row in candidates:
            try:
                sha256 = _compute_sha256(row["path"])
                db.update_sha256(row["id"], sha256)
            except Exception as e:
                db.update_error(row["id"], str(e))
                errors += 1

            batch_count += 1
            if batch_count >= batch_size:
                db.conn.commit()
                batch_count = 0

            progress.advance(task)

    db.conn.commit()
    console.print(f"[green]Done![/green] Hashed {len(candidates) - errors} images")
    if errors:
        console.print(f"[yellow]{errors} errors (see 'dedup status')[/yellow]")
