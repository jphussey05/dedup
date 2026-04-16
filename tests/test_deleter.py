"""Tests for deletion planning and execution."""

from __future__ import annotations

from pathlib import Path

from dedup.deleter import (
    _score_keeper,
    execute_deletions,
    export_plan,
    plan_deletions,
)


def _mk_row(
    path: str,
    file_size: int = 100,
    exif_date: str | None = None,
    file_mtime: str | None = None,
) -> dict:
    return {
        "id": 0,
        "path": path,
        "filename": Path(path).name,
        "file_size": file_size,
        "width": None,
        "height": None,
        "exif_date": exif_date,
        "file_mtime": file_mtime,
    }


# --- Scoring tests ---


def test_score_prefers_exif_over_everything():
    """EXIF beats depth, size, and mtime."""
    with_exif = _mk_row("a/b/c/d/e/f/deep.jpg", file_size=10, exif_date="2020:01:01 00:00:00")
    no_exif = _mk_row("shallow.jpg", file_size=9999, file_mtime="2000-01-01")
    assert _score_keeper(with_exif) < _score_keeper(no_exif)


def test_score_prefers_shallower_path_when_exif_tied():
    shallow = _mk_row("photos/a.jpg", file_size=100)
    deep = _mk_row("photos/backup/old/nested/a.jpg", file_size=100)
    assert _score_keeper(shallow) < _score_keeper(deep)


def test_score_prefers_larger_file_when_path_tied():
    small = _mk_row("a/b.jpg", file_size=100)
    large = _mk_row("a/b.jpg", file_size=1000)
    assert _score_keeper(large) < _score_keeper(small)


def test_score_prefers_older_mtime_as_tiebreaker():
    older = _mk_row("a/b.jpg", file_size=100, file_mtime="2005-01-01T00:00:00")
    newer = _mk_row("a/b.jpg", file_size=100, file_mtime="2020-01-01T00:00:00")
    assert _score_keeper(older) < _score_keeper(newer)


def test_score_handles_null_mtime():
    """Rows with no mtime should sort last, not crash."""
    has = _mk_row("a/b.jpg", file_mtime="2010-01-01")
    none = _mk_row("a/b.jpg", file_mtime=None)
    assert _score_keeper(has) < _score_keeper(none)


# --- Planning tests ---


def test_plan_skips_groups_with_one_member(db):
    """A group that only has one remaining image should not produce a decision."""
    db.insert_image("/only.jpg", "only.jpg", 100)
    db.insert_duplicate_group("exact", 1.0, 1)
    db.conn.commit()

    decisions = plan_deletions(db, match_type="exact")
    assert decisions == []


def test_plan_picks_exif_as_keeper(db):
    """EXIF wins even if the non-EXIF copy is larger and older."""
    id_exif = db.insert_image(
        "/deep/nested/exif.jpg", "exif.jpg", 100,
        exif_date="2020:01:01 00:00:00", file_mtime="2020-01-01T00:00:00",
    )
    id_plain = db.insert_image(
        "/plain.jpg", "plain.jpg", 9999,
        exif_date=None, file_mtime="2005-01-01T00:00:00",
    )
    group_id = db.insert_duplicate_group("exact", 1.0, 2)
    db.insert_duplicate_pair(group_id, id_exif, id_plain, 1.0)
    db.conn.commit()

    decisions = plan_deletions(db, match_type="exact")
    assert len(decisions) == 1
    assert decisions[0].keeper["id"] == id_exif
    assert len(decisions[0].deletions) == 1
    assert decisions[0].deletions[0]["id"] == id_plain


def test_plan_respects_match_type_filter(db):
    """match_type='exact' should ignore perceptual groups."""
    id1 = db.insert_image("/a.jpg", "a.jpg", 100)
    id2 = db.insert_image("/b.jpg", "b.jpg", 100)
    g_exact = db.insert_duplicate_group("exact", 1.0, 2)
    db.insert_duplicate_pair(g_exact, id1, id2, 1.0)

    id3 = db.insert_image("/c.jpg", "c.jpg", 100)
    id4 = db.insert_image("/d.jpg", "d.jpg", 100)
    g_perc = db.insert_duplicate_group("perceptual", 0.95, 2)
    db.insert_duplicate_pair(g_perc, id3, id4, 0.95)
    db.conn.commit()

    exact_only = plan_deletions(db, match_type="exact")
    assert len(exact_only) == 1
    assert exact_only[0].match_type == "exact"

    all_types = plan_deletions(db, match_type="all")
    assert len(all_types) == 2


def test_plan_respects_limit(db):
    """--limit caps the number of groups."""
    for i in range(5):
        a = db.insert_image(f"/g{i}/a.jpg", "a.jpg", 100)
        b = db.insert_image(f"/g{i}/b.jpg", "b.jpg", 100)
        g = db.insert_duplicate_group("exact", 1.0, 2)
        db.insert_duplicate_pair(g, a, b, 1.0)
    db.conn.commit()

    decisions = plan_deletions(db, match_type="exact", limit=2)
    assert len(decisions) == 2


# --- Export test ---


def test_export_plan_writes_csv(db, tmp_dir):
    id_keep = db.insert_image("/keep.jpg", "keep.jpg", 500, exif_date="2020:01:01 00:00:00")
    id_drop = db.insert_image("/drop.jpg", "drop.jpg", 500)
    g = db.insert_duplicate_group("exact", 1.0, 2)
    db.insert_duplicate_pair(g, id_keep, id_drop, 1.0)
    db.conn.commit()

    decisions = plan_deletions(db, match_type="exact")
    csv_path = tmp_dir / "plan.csv"
    export_plan(decisions, csv_path)

    content = csv_path.read_text(encoding="utf-8")
    assert "KEEP" in content
    assert "DELETE" in content
    assert "/keep.jpg" in content
    assert "/drop.jpg" in content


# --- Execution tests ---


def test_execute_hard_delete_removes_file_and_row(db, tmp_dir):
    """With --no-trash, files are unlinked and DB rows dropped."""
    keep = tmp_dir / "keep.jpg"
    drop = tmp_dir / "drop.jpg"
    keep.write_bytes(b"keep")
    drop.write_bytes(b"drop")

    id_keep = db.insert_image(
        str(keep), "keep.jpg", 4, exif_date="2020:01:01 00:00:00"
    )
    id_drop = db.insert_image(str(drop), "drop.jpg", 4)
    g = db.insert_duplicate_group("exact", 1.0, 2)
    db.insert_duplicate_pair(g, id_keep, id_drop, 1.0)
    db.conn.commit()

    decisions = plan_deletions(db, match_type="exact")
    from rich.console import Console
    deleted, errors, freed = execute_deletions(db, decisions, use_trash=False, console=Console())

    assert deleted == 1
    assert errors == 0
    assert freed == 4
    assert keep.exists()
    assert not drop.exists()

    # DB row for drop should be gone; keeper remains
    remaining = db.conn.execute(
        "SELECT id FROM images ORDER BY id"
    ).fetchall()
    assert [r["id"] for r in remaining] == [id_keep]

    # Empty group should have been cleaned up
    groups = db.conn.execute("SELECT COUNT(*) as n FROM duplicate_groups").fetchone()
    assert groups["n"] == 0


def test_execute_missing_file_still_cleans_db(db, tmp_dir):
    """If a file is already gone on disk, the DB row should still be removed."""
    keep = tmp_dir / "keep.jpg"
    keep.write_bytes(b"keep")
    missing = tmp_dir / "missing.jpg"  # never created

    id_keep = db.insert_image(
        str(keep), "keep.jpg", 4, exif_date="2020:01:01 00:00:00"
    )
    id_missing = db.insert_image(str(missing), "missing.jpg", 4)
    g = db.insert_duplicate_group("exact", 1.0, 2)
    db.insert_duplicate_pair(g, id_keep, id_missing, 1.0)
    db.conn.commit()

    decisions = plan_deletions(db, match_type="exact")
    from rich.console import Console
    deleted, errors, _ = execute_deletions(
        db, decisions, use_trash=False, console=Console()
    )

    # FileNotFoundError is swallowed; no "error", just still cleans DB
    assert errors == 0
    remaining = db.conn.execute(
        "SELECT id FROM images ORDER BY id"
    ).fetchall()
    assert [r["id"] for r in remaining] == [id_keep]
