"""Deletion planning and execution for duplicate groups.

Scoring priority (highest first) for picking the keeper in each group:
    1. EXIF data present (some copy processes strip it — EXIF is the strongest
       signal of an original).
    2. Shortest/shallowest path (buried copies are usually backups).
    3. Largest file size (higher quality for perceptual/CNN matches).
    4. Oldest file_mtime (tiebreaker — originals tend to be older).
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from dedup.db import Database


@dataclass
class GroupDecision:
    group_id: int
    match_type: str
    confidence: float
    keeper: dict
    deletions: list[dict] = field(default_factory=list)


def _score_keeper(row: sqlite3.Row | dict) -> tuple:
    """Sort key for picking the best keeper. Smaller tuple = better keeper."""
    path = row["path"] or ""
    has_exif = row["exif_date"] is not None
    path_depth = path.count("\\") + path.count("/")
    file_size = row["file_size"] or 0
    mtime = row["file_mtime"] or "9999"  # missing mtime sorts last

    return (
        not has_exif,   # False (has EXIF) < True — has_exif wins
        path_depth,     # shallower wins
        -file_size,     # larger wins
        mtime,          # older ISO date wins (lexicographic)
    )


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def plan_deletions(
    db: Database,
    match_type: str = "exact",
    min_confidence: float = 0.0,
    limit: int | None = None,
) -> list[GroupDecision]:
    """Walk duplicate groups and pick a keeper + deletions for each."""
    groups = db.get_duplicate_groups(match_type, min_confidence)
    if limit is not None:
        groups = groups[:limit]

    decisions: list[GroupDecision] = []
    for group in groups:
        members = [_row_to_dict(r) for r in db.get_group_members(group["id"])]
        if len(members) < 2:
            continue

        sorted_members = sorted(members, key=_score_keeper)
        keeper = sorted_members[0]
        deletions = sorted_members[1:]

        decisions.append(
            GroupDecision(
                group_id=group["id"],
                match_type=group["match_type"],
                confidence=group["confidence"],
                keeper=keeper,
                deletions=deletions,
            )
        )
    return decisions


def _format_size(size: int | None) -> str:
    if not size:
        return "0 B"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def _keeper_reason(keeper: dict, deletions: list[dict]) -> str:
    """One-word reason this row won — for the preview table."""
    if keeper["exif_date"] is not None and any(d["exif_date"] is None for d in deletions):
        return "EXIF"
    keeper_depth = (keeper["path"] or "").count("\\") + (keeper["path"] or "").count("/")
    del_depths = [(d["path"] or "").count("\\") + (d["path"] or "").count("/") for d in deletions]
    if all(keeper_depth < d for d in del_depths):
        return "shallower"
    if all((keeper["file_size"] or 0) >= (d["file_size"] or 0) for d in deletions):
        if any((keeper["file_size"] or 0) > (d["file_size"] or 0) for d in deletions):
            return "larger"
    return "oldest"


def preview_deletions(
    decisions: list[GroupDecision], console: Console, max_groups: int = 20
) -> None:
    """Print a Rich table summarizing the deletion plan."""
    if not decisions:
        console.print("[dim]No duplicate groups to review.[/dim]")
        return

    total_deletions = sum(len(d.deletions) for d in decisions)
    total_freed = sum(sum(x["file_size"] or 0 for x in d.deletions) for d in decisions)

    table = Table(title=f"Deletion preview (first {min(max_groups, len(decisions))} groups)")
    table.add_column("Group", style="cyan", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Path")
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("EXIF", justify="center", no_wrap=True)
    table.add_column("Reason", no_wrap=True)

    for decision in decisions[:max_groups]:
        reason = _keeper_reason(decision.keeper, decision.deletions)
        table.add_row(
            str(decision.group_id),
            decision.match_type,
            "[green]KEEP[/green]",
            decision.keeper["path"],
            _format_size(decision.keeper["file_size"]),
            "✓" if decision.keeper["exif_date"] else "-",
            reason,
        )
        for d in decision.deletions:
            table.add_row(
                "",
                "",
                "[red]DELETE[/red]",
                d["path"],
                _format_size(d["file_size"]),
                "✓" if d["exif_date"] else "-",
                "",
            )
        table.add_row("", "", "", "", "", "", "")

    console.print(table)
    console.print(
        f"[bold]Total:[/bold] {len(decisions)} groups, {total_deletions} files to delete, "
        f"~{_format_size(total_freed)} to free"
    )
    if len(decisions) > max_groups:
        console.print(f"[dim]({len(decisions) - max_groups} more groups not shown)[/dim]")


def export_plan(decisions: list[GroupDecision], path: Path) -> None:
    """Write the deletion plan to a CSV for offline review."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group_id", "match_type", "confidence", "role",
            "path", "file_size", "exif_date", "file_mtime",
        ])
        for d in decisions:
            writer.writerow([
                d.group_id, d.match_type, f"{d.confidence:.4f}", "KEEP",
                d.keeper["path"], d.keeper["file_size"],
                d.keeper["exif_date"] or "", d.keeper["file_mtime"] or "",
            ])
            for victim in d.deletions:
                writer.writerow([
                    d.group_id, d.match_type, f"{d.confidence:.4f}", "DELETE",
                    victim["path"], victim["file_size"],
                    victim["exif_date"] or "", victim["file_mtime"] or "",
                ])


def execute_deletions(
    db: Database,
    decisions: list[GroupDecision],
    use_trash: bool,
    console: Console,
) -> tuple[int, int, int]:
    """Actually delete files and clean up DB.

    Returns (deleted_count, error_count, freed_bytes).
    """
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    deleter_fn = _trash_file if use_trash else _hard_delete_file

    deleted_count = 0
    error_count = 0
    freed_bytes = 0
    deleted_ids: list[int] = []

    total = sum(len(d.deletions) for d in decisions)
    if total == 0:
        console.print("[dim]Nothing to delete.[/dim]")
        return 0, 0, 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Deleting", total=total)
        for decision in decisions:
            for victim in decision.deletions:
                path = victim["path"]
                size = victim["file_size"] or 0
                try:
                    deleter_fn(path)
                    deleted_count += 1
                    freed_bytes += size
                    deleted_ids.append(victim["id"])
                except FileNotFoundError:
                    # File already gone — still drop from DB
                    deleted_ids.append(victim["id"])
                except Exception as exc:
                    error_count += 1
                    console.print(f"[red]Failed:[/red] {path} ({exc})")
                progress.advance(task)

    if deleted_ids:
        db.delete_images(deleted_ids)
        stale = db.cleanup_stale_groups()
        db.conn.commit()
        if stale:
            console.print(f"[dim]Removed {stale} now-empty duplicate groups.[/dim]")

    return deleted_count, error_count, freed_bytes


def _trash_file(path: str) -> None:
    from send2trash import send2trash

    send2trash(path)


def _hard_delete_file(path: str) -> None:
    Path(path).unlink()
