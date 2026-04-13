"""Report generation for duplicate findings."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from dedup.db import Database


def generate_report(
    db: Database,
    format: str = "table",
    output: Path | None = None,
    match_type: str = "all",
    min_confidence: float = 0.0,
) -> None:
    """Generate a report of found duplicates."""
    console = Console()

    rows = db.get_duplicate_report(match_type, min_confidence)

    if not rows:
        console.print("[dim]No duplicates found matching the criteria.[/dim]")
        return

    if format == "table":
        _render_table(console, rows)
    elif format == "csv":
        _render_csv(rows, output)
        if output:
            console.print(f"[green]CSV written to {output}[/green]")
    elif format == "json":
        _render_json(rows, output)
        if output:
            console.print(f"[green]JSON written to {output}[/green]")
    else:
        console.print(f"[red]Unknown format: {format}[/red]")


def _render_table(console: Console, rows: list[dict]) -> None:
    """Render duplicate report as a Rich table."""
    table = Table(title="Duplicate Report", show_lines=True)
    table.add_column("Group", style="bold", width=6)
    table.add_column("Type", width=12)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("File 1", max_width=50)
    table.add_column("Size 1", justify="right", width=10)
    table.add_column("File 2", max_width=50)
    table.add_column("Size 2", justify="right", width=10)

    for row in rows:
        size1 = _format_size(row["size1"])
        size2 = _format_size(row["size2"])
        sim = f"{row['similarity']:.2%}" if row["match_type"] != "exact" else "100%"
        type_style = {
            "exact": "[red]exact[/red]",
            "perceptual": "[yellow]perceptual[/yellow]",
            "cnn": "[cyan]cnn[/cyan]",
        }.get(row["match_type"], row["match_type"])

        table.add_row(
            str(row["group_id"]),
            type_style,
            sim,
            row["path1"],
            size1,
            row["path2"],
            size2,
        )

    console.print(table)
    console.print(f"\n[bold]{len(rows)} duplicate pairs shown[/bold]")


def _render_csv(rows: list[dict], output: Path | None) -> None:
    """Render duplicate report as CSV."""
    fieldnames = [
        "group_id", "match_type", "similarity",
        "path1", "size1", "name1",
        "path2", "size2", "name2",
    ]

    f = open(output, "w", newline="", encoding="utf-8") if output else sys.stdout
    try:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if output and f is not sys.stdout:
            f.close()


def _render_json(rows: list[dict], output: Path | None) -> None:
    """Render duplicate report as JSON."""
    # Group rows by group_id for a cleaner structure
    groups: dict[int, dict] = {}
    for row in rows:
        gid = row["group_id"]
        if gid not in groups:
            groups[gid] = {
                "group_id": gid,
                "match_type": row["match_type"],
                "confidence": row["confidence"],
                "pairs": [],
            }
        groups[gid]["pairs"].append({
            "path1": row["path1"],
            "size1": row["size1"],
            "path2": row["path2"],
            "size2": row["size2"],
            "similarity": row["similarity"],
        })

    data = list(groups.values())
    text = json.dumps(data, indent=2)

    if output:
        output.write_text(text, encoding="utf-8")
    else:
        print(text)


def generate_error_report(
    db: Database,
    limit: int = 50,
    stage: str = "all",
    export: Path | None = None,
) -> None:
    """Show files that failed processing and why."""
    console = Console()

    rows = db.get_errored_images(stage=stage, limit=limit)

    if not rows:
        console.print("[green]No errors found.[/green]")
        return

    # Group errors by message to show common patterns first
    from collections import Counter
    error_counts = Counter(r["error"] for r in rows)

    console.print(f"\n[bold red]{len(rows)} errored files[/bold red] (showing up to {limit})\n")

    # Show top error patterns
    console.print("[bold]Most common errors:[/bold]")
    for msg, count in error_counts.most_common(5):
        short = msg[:80] + "..." if len(msg) > 80 else msg
        console.print(f"  [yellow]{count}x[/yellow] {short}")

    console.print()

    # Show individual files
    table = Table(show_lines=False, box=None)
    table.add_column("File", max_width=60)
    table.add_column("Size", justify="right", width=10)
    table.add_column("Error", max_width=60)

    for row in rows:
        size_str = _format_size(row["file_size"])
        error = row["error"] or ""
        short_error = error[:60] + "..." if len(error) > 60 else error
        table.add_row(row["path"], size_str, f"[yellow]{short_error}[/yellow]")

    console.print(table)

    if export:
        import csv as csv_mod
        with open(export, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(
                f,
                fieldnames=["id", "path", "filename", "file_size", "error"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"\n[green]Exported to {export}[/green]")


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    elif size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"
