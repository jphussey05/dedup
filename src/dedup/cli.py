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
    workers: int = typer.Option(0, help="CPU workers (0 = auto: cpu_count - 1)"),
) -> None:
    """Compute perceptual hashes and find near-duplicate images."""
    from dedup.comparator import find_perceptual_duplicates
    from dedup.perceptual import compute_perceptual_hashes

    database = get_db(db)
    try:
        compute_perceptual_hashes(database, workers=workers if workers > 0 else None)
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
def review(
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
    match_type: str = typer.Option(
        "exact",
        help="Which groups to process: exact, perceptual, cnn, or all. "
             "Default 'exact' is the only risk-free auto-delete (SHA256 match).",
    ),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence score"),
    export: Path | None = typer.Option(
        None, "--export", "-o", help="Write deletion plan to CSV"
    ),
    execute: bool = typer.Option(
        False, "--execute", help="Actually delete files (otherwise preview only)"
    ),
    use_trash: bool = typer.Option(
        True,
        "--trash/--no-trash",
        help="Send files to Recycle Bin (default) or permanently delete",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt when executing"
    ),
    limit: int | None = typer.Option(
        None, help="Only process the first N groups (useful for testing)"
    ),
) -> None:
    """Pick a keeper per duplicate group and preview or delete the rest.

    Scoring priority: EXIF present > shallowest path > largest size > oldest mtime.
    Default is preview only; pass --execute to delete.
    """
    from dedup.deleter import execute_deletions, export_plan, plan_deletions, preview_deletions

    database = get_db(db)
    try:
        decisions = plan_deletions(database, match_type, min_confidence, limit)

        if not decisions:
            console.print("[dim]No duplicate groups matched the filter.[/dim]")
            return

        preview_deletions(decisions, console)

        if export:
            export_plan(decisions, export)
            console.print(f"[green]Wrote deletion plan to {export}[/green]")

        if not execute:
            console.print(
                "\n[dim]Preview only. Re-run with [bold]--execute[/bold] to delete.[/dim]"
            )
            return

        total = sum(len(d.deletions) for d in decisions)
        mode = "Recycle Bin" if use_trash else "[red]PERMANENT DELETE[/red]"
        console.print(f"\nAbout to delete [bold]{total}[/bold] files via {mode}.")

        if not yes:
            confirm = typer.confirm("Proceed?", default=False)
            if not confirm:
                console.print("[yellow]Aborted.[/yellow]")
                return

        deleted, errs, freed = execute_deletions(database, decisions, use_trash, console)
        freed_mb = freed / 1024 / 1024
        console.print(
            f"[green]Deleted {deleted} files[/green] "
            f"({freed_mb:.1f} MB freed){f', [red]{errs} errors[/red]' if errs else ''}"
        )
    finally:
        database.close()


@app.command()
def organize(
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
    dest: str = typer.Option(
        "//hussey_nas/hussey share/photos_organized",
        "--dest",
        help="Destination root for the consolidated archive",
    ),
    execute: bool = typer.Option(
        False, "--execute", help="Actually move files (otherwise preview only)"
    ),
    export: Path | None = typer.Option(
        None, "--export", "-o", help="Write move plan to CSV"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt when executing"
    ),
    limit: int | None = typer.Option(
        None, help="Only process the first N images (useful for testing)"
    ),
    exclude_source: list[str] = typer.Option(
        [],
        "--exclude-source",
        help=(
            "Source path prefix to skip; repeatable. Any row whose current "
            "path starts with this prefix is left where it is. Example: "
            "--exclude-source '//hussey_nas/hussey share/whitney pro photos'"
        ),
    ),
) -> None:
    """Rename & move surviving images into dest/YYYY/YYYY-MM-DD_HHMMSS.ext.

    Run this AFTER ``dedup review --execute`` has removed duplicates. Uses EXIF
    DateTimeOriginal when available, falls back to file mtime. Preview-only
    unless --execute is passed.
    """
    from dedup.organizer import execute_moves, export_plan, plan_moves, preview_moves

    database = get_db(db)
    try:
        # Heads-up: if there are unresolved exact duplicate groups, the user
        # almost certainly wants to run `review --execute` first.
        remaining = database.get_duplicate_groups("exact")
        if remaining:
            console.print(
                f"[yellow]Warning:[/yellow] {len(remaining)} unresolved exact duplicate "
                "groups still in DB. Consider running [bold]dedup review --execute[/bold] first."
            )

        dest_root = Path(dest)
        decisions, skipped = plan_moves(
            database, dest_root, limit,
            console=console,
            exclude_sources=exclude_source,
        )

        if not decisions and not skipped:
            console.print("[dim]No surviving images to organize.[/dim]")
            return

        preview_moves(decisions, console, skipped_no_date=skipped)

        if export:
            export_plan(decisions, export)
            console.print(f"[green]Wrote move plan to {export}[/green]")

        if not decisions:
            return

        if not execute:
            console.print(
                "\n[dim]Preview only. Re-run with [bold]--execute[/bold] to move.[/dim]"
            )
            return

        console.print(
            f"\nAbout to move [bold]{len(decisions)}[/bold] files into "
            f"[bold]{dest_root}[/bold]."
        )

        if not yes:
            confirm = typer.confirm("Proceed?", default=False)
            if not confirm:
                console.print("[yellow]Aborted.[/yellow]")
                return

        moved, errs = execute_moves(database, decisions, console)
        console.print(
            f"[green]Moved {moved} files[/green]"
            f"{f', [red]{errs} errors[/red]' if errs else ''}"
        )
    finally:
        database.close()


@app.command()
def errors(
    db: Path = typer.Option(Path("dedup.db"), "--db", help="SQLite database path"),
    limit: int = typer.Option(50, help="Max errors to show"),
    stage: str = typer.Option("all", help="Filter by stage: scan, hash, phash, embed, all"),
    export: Path | None = typer.Option(None, "--export", "-o", help="Export error list to CSV"),
) -> None:
    """Show files that failed processing and why."""
    from dedup.reporter import generate_error_report

    database = get_db(db)
    try:
        generate_error_report(database, limit, stage, export)
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
