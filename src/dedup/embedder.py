"""Stage 3: CNN embedding generation using CLIP via sentence-transformers."""

from __future__ import annotations

import numpy as np
from PIL import Image
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from dedup.db import Database

Image.MAX_IMAGE_PIXELS = 500_000_000


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to cuda or cpu."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def compute_embeddings(
    db: Database,
    model_name: str = "clip-ViT-B-32",
    batch_size: int = 64,
    device: str = "auto",
) -> None:
    """Compute CLIP embeddings for all unprocessed images."""
    from rich.console import Console
    from sentence_transformers import SentenceTransformer

    console = Console()
    resolved_device = _resolve_device(device)
    console.print(f"[bold]Loading model:[/bold] {model_name} on {resolved_device}")

    model = SentenceTransformer(model_name, device=resolved_device)

    images = db.get_unembedded_images()
    if not images:
        console.print("[green]All images already have embeddings.[/green]")
        return

    console.print(f"[bold]Computing embeddings for {len(images)} images[/bold]")

    errors = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Generating embeddings", total=len(images))

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            pil_images = []
            valid_rows = []

            for row in batch:
                try:
                    img = Image.open(row["path"]).convert("RGB")
                    pil_images.append(img)
                    valid_rows.append(row)
                except Exception as e:
                    db.update_error(row["id"], str(e))
                    errors += 1
                    progress.advance(task)

            if pil_images:
                try:
                    embeddings = model.encode(
                        pil_images,
                        batch_size=len(pil_images),
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )

                    for row, emb in zip(valid_rows, embeddings):
                        blob = np.array(emb, dtype=np.float32).tobytes()
                        db.update_embedding(row["id"], blob)
                        progress.advance(task)

                except Exception as e:
                    for row in valid_rows:
                        db.update_error(row["id"], str(e))
                        errors += 1
                        progress.advance(task)

                # Close PIL images to free memory
                for img in pil_images:
                    img.close()

            db.conn.commit()

    console.print(f"[green]Done![/green] Computed embeddings for {len(images) - errors} images")
    if errors:
        console.print(f"[yellow]{errors} errors[/yellow]")
