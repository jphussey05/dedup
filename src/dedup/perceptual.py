"""Stage 2: Perceptual hash computation (pHash + dHash)."""

from __future__ import annotations

import multiprocessing as mp
import os

import imagehash
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from dedup.db import Database

Image.MAX_IMAGE_PIXELS = 500_000_000


def _to_signed_int64(val: int) -> int:
    """Convert an unsigned 64-bit integer to signed for SQLite storage."""
    if val >= (1 << 63):
        val -= 1 << 64
    return val


def _from_signed_int64(val: int) -> int:
    """Convert a signed 64-bit integer back to unsigned for Hamming distance."""
    if val < 0:
        val += 1 << 64
    return val


def _compute_hashes_worker(
    args: tuple[int, str],
) -> tuple[int, str, int, str, int] | tuple[int, None, None, None, None, str]:
    """Worker function: compute pHash + dHash for a single image.

    Runs in a subprocess. Returns (id, ph_hex, ph_int, dh_hex, dh_int)
    or (id, None, None, None, None, error_message) on failure.
    """
    image_id, path = args
    Image.MAX_IMAGE_PIXELS = 500_000_000
    try:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            ph = imagehash.phash(img_rgb, hash_size=8)
            dh = imagehash.dhash(img_rgb, hash_size=8)

        ph_hex = str(ph)
        dh_hex = str(dh)
        ph_int = _to_signed_int64(int(ph_hex, 16))
        dh_int = _to_signed_int64(int(dh_hex, 16))

        return (image_id, ph_hex, ph_int, dh_hex, dh_int)  # type: ignore[return-value]
    except Exception as e:
        return (image_id, None, None, None, None, str(e))  # type: ignore[return-value]


def compute_perceptual_hashes(
    db: Database,
    workers: int | None = None,
) -> None:
    """Compute perceptual hashes using multiple CPU cores."""
    from rich.console import Console

    console = Console()

    images = db.get_unphashed_images()
    if not images:
        console.print("[green]All images already have perceptual hashes.[/green]")
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)

    total = len(images)
    console.print(
        f"[bold]Computing perceptual hashes for {total} images[/bold] "
        f"using {workers} workers"
    )

    args = [(row["id"], row["path"]) for row in images]

    errors = 0
    batch: list[tuple] = []
    batch_size = 500

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[bold]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Perceptual hashing", total=total)

        # Use Pool.imap_unordered — streams results as workers finish,
        # no 317K Future objects in memory, progress updates immediately.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(
                _compute_hashes_worker, args, chunksize=32
            ):
                if len(result) == 6:
                    image_id, _, _, _, _, error_msg = result
                    db.update_error(image_id, error_msg)
                    errors += 1
                else:
                    image_id, ph_hex, ph_int, dh_hex, dh_int = result
                    batch.append((image_id, ph_hex, ph_int, dh_hex, dh_int))

                if len(batch) >= batch_size:
                    _flush_batch(db, batch)
                    batch.clear()

                progress.advance(task)

    if batch:
        _flush_batch(db, batch)

    db.conn.commit()
    console.print(
        f"[green]Done![/green] Computed hashes for {total - errors} images"
    )
    if errors:
        console.print(
            f"[yellow]{errors} errors — run 'dedup errors' for details[/yellow]"
        )


def _flush_batch(db: Database, batch: list[tuple]) -> None:
    """Write a batch of phash results to the database."""
    for image_id, ph_hex, ph_int, dh_hex, dh_int in batch:
        db.update_phash(image_id, ph_hex, ph_int, dh_hex, dh_int)
    db.conn.commit()
