"""Stage 2: Perceptual hash computation (pHash + dHash)."""

from __future__ import annotations

import imagehash
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from dedup.db import Database


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


def _compute_hashes(path: str) -> tuple[str, int, str, int]:
    """Compute pHash and dHash for an image.

    Returns (phash_hex, phash_int_signed, dhash_hex, dhash_int_signed).
    """
    with Image.open(path) as img:
        ph = imagehash.phash(img, hash_size=8)
        dh = imagehash.dhash(img, hash_size=8)

    ph_hex = str(ph)
    dh_hex = str(dh)
    ph_int = _to_signed_int64(int(ph_hex, 16))
    dh_int = _to_signed_int64(int(dh_hex, 16))

    return ph_hex, ph_int, dh_hex, dh_int


def compute_perceptual_hashes(db: Database) -> None:
    """Compute perceptual hashes for all unprocessed images."""
    from rich.console import Console

    console = Console()

    images = db.get_unphashed_images()
    if not images:
        console.print("[green]All images already have perceptual hashes.[/green]")
        return

    console.print(f"[bold]Computing perceptual hashes for {len(images)} images[/bold]")

    errors = 0
    batch_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Perceptual hashing", total=len(images))

        for row in images:
            try:
                ph_hex, ph_int, dh_hex, dh_int = _compute_hashes(row["path"])
                db.update_phash(row["id"], ph_hex, ph_int, dh_hex, dh_int)
            except Exception as e:
                db.update_error(row["id"], str(e))
                errors += 1

            batch_count += 1
            if batch_count >= 500:
                db.conn.commit()
                batch_count = 0

            progress.advance(task)

    db.conn.commit()
    console.print(f"[green]Done![/green] Computed hashes for {len(images) - errors} images")
    if errors:
        console.print(f"[yellow]{errors} errors[/yellow]")
