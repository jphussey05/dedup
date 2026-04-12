"""CLI entry point for dedup."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dedup.db import Database

app = typer.Typer(
    name="dedup",
    help="Find duplicate and similar images across directories and NAS backups.",
    no_args_is_help=True,
)
console = Console()


def get_db(db: Path) -> Database:
    database = Database(db)
    database.init_schema()
    return database


@app.command()
def scan(
    path: str = typer.Argument(..., help="Directory path to scan for images"),
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
    recursive: bool = typer.Option(True, help="Scan subdirectories recursively"),
    formats: str = typer.Option(
        ".jpg,.jpeg,.png,.tiff,.tif,.bmp,.heic,.heif,.webp",
        help="Comma-separated image file extensions to include",
    ),
) -> None:
    """Discover images in a directory and store metadata in the database."""
    from dedup.scanner import scan_directory

    database = get_db(db)
    try:
        format_set = {f.strip().lower() for f in formats.split(",")}
        scan_directory(database, path, format_set, recursive)
    finally:
        database.close()


@app.command()
def hash(
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
    batch_size: int = typer.Option(1000, help="Commit batch size"),
) -> None:
    """Compute SHA256 hashes and find exact duplicates."""
    from dedup.comparator import find_exact_duplicates
    from dedup.hasher import hash_images

    database = get_db(db)
    try:
        hash_images(database, batch_size)
        find_exact_duplicates(database)
    finally:
        database.close()


@app.command()
def phash(
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
    algorithm: str = typer.Option("both", help="Hash algorithm: phash, dhash, or both"),
    threshold: int = typer.Option(10, help="Max Hamming distance for a match"),
) -> None:
    """Compute perceptual hashes and find near-duplicate images."""
    from dedup.comparator import find_perceptual_duplicates
    from dedup.perceptual import compute_perceptual_hashes

    database = get_db(db)
    try:
        compute_perceptual_hashes(database)
        find_perceptual_duplicates(database, threshold)
    finally:
        database.close()


@app.command()
def compare(
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
    model: str = typer.Option("clip-ViT-B-32", help="Sentence-transformers model name"),
    threshold: float = typer.Option(0.92, help="Minimum cosine similarity for a match"),
    batch_size: int = typer.Option(64, help="Batch size for embedding generation"),
    device: str = typer.Option("auto", help="Device: cuda, cpu, or auto"),
) -> None:
    """Generate CNN embeddings and find semantically similar images."""
    try:
        from dedup.comparator import find_cnn_duplicates
        from dedup.embedder import compute_embeddings
    except ImportError:
        console.print(
            "[red]CNN comparison requires extra dependencies.[/red]\n"
            "Install with: [bold]uv sync --extra cnn[/bold]"
        )
        raise typer.Exit(1)

    database = get_db(db)
    try:
        compute_embeddings(database, model, batch_size, device)
        find_cnn_duplicates(database, threshold)
    finally:
        database.close()


@app.command()
def report(
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
    format: str = typer.Option("table", help="Output format: table, csv, or json"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file path"),
    match_type: str = typer.Option("all", help="Filter: exact, perceptual, cnn, or all"),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence score"),
) -> None:
    """Generate a report of found duplicates."""
    from dedup.reporter import generate_report

    database = get_db(db)
    try:
        generate_report(database, format, output, match_type, min_confidence)
    finally:
        database.close()


@app.command()
def status(
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
) -> None:
    """Show processing status and progress."""
    database = get_db(db)
    try:
        stats = database.get_status()
    finally:
        database.close()

    total = stats["total_images"]

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total images", str(total))
    table.add_row("SHA256 hashed", f"{stats['hashed']}/{total}")
    table.add_row("Perceptual hashed", f"{stats['phashed']}/{total}")
    table.add_row("CNN embedded", f"{stats['embedded']}/{total}")
    table.add_row("Errors", str(stats["errored"]))
    table.add_row("", "")
    table.add_row("Exact duplicate groups", str(stats["exact_groups"]))
    table.add_row("Perceptual duplicate groups", str(stats["perceptual_groups"]))
    table.add_row("CNN duplicate groups", str(stats["cnn_groups"]))

    db_size = db.stat().st_size if db.exists() else 0
    if db_size > 1024 * 1024:
        size_str = f"{db_size / 1024 / 1024:.1f} MB"
    else:
        size_str = f"{db_size / 1024:.1f} KB"
    table.add_row("", "")
    table.add_row("Database size", size_str)

    console.print(Panel(table, title="[bold]dedup status[/bold]", border_style="blue"))
