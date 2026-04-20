"""Tests for the organize command — date parsing, collision-safe naming, moves."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.console import Console

from dedup.organizer import (
    _build_dest,
    _parse_date,
    execute_moves,
    export_plan,
    plan_moves,
)
from dedup.pathutil import normalize_path

# --- Date parsing ---


def test_parse_date_prefers_exif():
    dt, src = _parse_date("2020:06:15 14:30:00", "2005-01-01T00:00:00")
    assert src == "exif"
    assert dt == datetime(2020, 6, 15, 14, 30, 0)


def test_parse_date_falls_back_to_mtime():
    dt, src = _parse_date(None, "2010-03-04T05:06:07")
    assert src == "mtime"
    assert dt == datetime(2010, 3, 4, 5, 6, 7)


def test_parse_date_handles_malformed_exif():
    """Zeroed-out EXIF is common on phone cameras — should fall through to mtime."""
    dt, src = _parse_date("0000:00:00 00:00:00", "2015-07-20T09:00:00")
    assert src == "mtime"
    assert dt == datetime(2015, 7, 20, 9, 0, 0)


def test_parse_date_returns_none_when_both_missing():
    dt, src = _parse_date(None, None)
    assert dt is None
    assert src == "none"


def test_parse_date_returns_none_when_both_garbage():
    dt, src = _parse_date("not a date", "also garbage")
    assert dt is None
    assert src == "none"


# --- Destination naming ---


def test_build_dest_basic(tmp_dir):
    dt = datetime(2021, 5, 10, 12, 34, 56)
    dest = _build_dest(dt, ".jpg", tmp_dir, taken=set())
    assert dest == tmp_dir / "2021" / "2021-05-10_123456.jpg"


def test_build_dest_collision_suffix(tmp_dir):
    """Two images with the same timestamp get _001 / _002 suffixes."""
    dt = datetime(2021, 5, 10, 12, 34, 56)
    taken: set[Path] = set()
    d1 = _build_dest(dt, ".jpg", tmp_dir, taken)
    taken.add(d1)
    d2 = _build_dest(dt, ".jpg", tmp_dir, taken)
    taken.add(d2)
    d3 = _build_dest(dt, ".jpg", tmp_dir, taken)
    assert d1.name == "2021-05-10_123456.jpg"
    assert d2.name == "2021-05-10_123456_001.jpg"
    assert d3.name == "2021-05-10_123456_002.jpg"


def test_build_dest_respects_occupied_set(tmp_dir):
    """Pre-seeded `taken` entries (the on-disk snapshot) should force a suffix bump."""
    dt = datetime(2021, 5, 10, 12, 34, 56)
    occupied: set[Path] = {tmp_dir / "2021" / "2021-05-10_123456.jpg"}
    dest = _build_dest(dt, ".jpg", tmp_dir, taken=occupied)
    assert dest.name == "2021-05-10_123456_001.jpg"


def test_snapshot_dest_picks_up_existing_files(tmp_dir):
    """_snapshot_dest returns every file in every YYYY/ subdir — no stat per image needed."""
    from dedup.organizer import _snapshot_dest

    year_dir = tmp_dir / "2021"
    year_dir.mkdir()
    (year_dir / "a.jpg").write_bytes(b"x")
    (year_dir / "b.jpg").write_bytes(b"x")
    (tmp_dir / "2022").mkdir()
    (tmp_dir / "2022" / "c.jpg").write_bytes(b"x")

    snap = _snapshot_dest(tmp_dir)
    assert snap == {
        year_dir / "a.jpg",
        year_dir / "b.jpg",
        tmp_dir / "2022" / "c.jpg",
    }


def test_snapshot_dest_on_missing_root_returns_empty(tmp_dir):
    from dedup.organizer import _snapshot_dest
    assert _snapshot_dest(tmp_dir / "does_not_exist") == set()


def test_build_dest_lowercases_extension(tmp_dir):
    dt = datetime(2020, 1, 1, 0, 0, 0)
    dest = _build_dest(dt, ".JPG", tmp_dir, taken=set())
    assert dest.suffix == ".jpg"


# --- Planning ---


def test_plan_moves_uses_exif_when_present(db, tmp_dir):
    src = tmp_dir / "random.jpg"
    src.write_bytes(b"x")
    db.insert_image(
        normalize_path(src), "random.jpg", 1,
        exif_date="2019:08:14 10:20:30",
        file_mtime="2005-01-01T00:00:00",
    )
    db.conn.commit()

    dest_root = tmp_dir / "archive"
    decisions, skipped = plan_moves(db, dest_root)
    assert skipped == 0
    assert len(decisions) == 1
    assert decisions[0].date_source == "exif"
    assert decisions[0].dest_year == "2019"
    assert decisions[0].dest_path.endswith("2019-08-14_102030.jpg")


def test_plan_moves_skips_images_without_any_date(db, tmp_dir):
    src = tmp_dir / "nodate.jpg"
    src.write_bytes(b"x")
    db.insert_image(
        normalize_path(src), "nodate.jpg", 1,
        exif_date=None, file_mtime=None,
    )
    db.conn.commit()

    decisions, skipped = plan_moves(db, tmp_dir / "archive")
    assert decisions == []
    assert skipped == 1


def test_plan_moves_respects_limit(db, tmp_dir):
    for i in range(5):
        p = tmp_dir / f"img{i}.jpg"
        p.write_bytes(b"x")
        db.insert_image(
            normalize_path(p), f"img{i}.jpg", 1,
            exif_date=f"2020:01:0{i+1} 00:00:00",
        )
    db.conn.commit()

    decisions, _ = plan_moves(db, tmp_dir / "archive", limit=3)
    assert len(decisions) == 3


def test_plan_moves_skips_already_organized(db, tmp_dir):
    """If a row's current path already equals its computed destination, skip it."""
    dest_root = tmp_dir / "archive"
    dest_root.mkdir()
    # Simulate a file that's already sitting at the destination a prior run
    # would have chosen.
    year_dir = dest_root / "2020"
    year_dir.mkdir()
    existing = year_dir / "2020-01-01_120000.jpg"
    existing.write_bytes(b"x")

    db.insert_image(
        normalize_path(existing), existing.name, 1,
        exif_date="2020:01:01 12:00:00",
    )
    db.conn.commit()

    decisions, skipped = plan_moves(db, dest_root)
    assert decisions == []
    assert skipped == 0


def test_plan_moves_excludes_errored_rows(db, tmp_dir):
    src = tmp_dir / "bad.jpg"
    src.write_bytes(b"x")
    img_id = db.insert_image(
        normalize_path(src), "bad.jpg", 1,
        exif_date="2020:01:01 00:00:00",
    )
    db.update_error(img_id, "corrupt")
    db.conn.commit()

    decisions, _ = plan_moves(db, tmp_dir / "archive")
    assert decisions == []


def test_plan_moves_handles_timestamp_collisions(db, tmp_dir):
    """Two images with the same EXIF timestamp get distinct destinations."""
    for i in range(2):
        p = tmp_dir / f"twin{i}.jpg"
        p.write_bytes(b"x")
        db.insert_image(
            normalize_path(p), f"twin{i}.jpg", 1,
            exif_date="2020:01:01 12:00:00",
        )
    db.conn.commit()

    decisions, _ = plan_moves(db, tmp_dir / "archive")
    assert len(decisions) == 2
    dests = {Path(d.dest_path).name for d in decisions}
    assert dests == {"2020-01-01_120000.jpg", "2020-01-01_120000_001.jpg"}


# --- Export ---


def test_export_plan_writes_csv(db, tmp_dir):
    src = tmp_dir / "a.jpg"
    src.write_bytes(b"x")
    db.insert_image(
        normalize_path(src), "a.jpg", 1,
        exif_date="2020:01:01 00:00:00",
    )
    db.conn.commit()

    decisions, _ = plan_moves(db, tmp_dir / "archive")
    csv_path = tmp_dir / "plan.csv"
    export_plan(decisions, csv_path)

    content = csv_path.read_text(encoding="utf-8")
    assert "image_id,src_path,dest_path,date_source,dest_year,file_size" in content
    assert "exif" in content
    assert "2020" in content


# --- Execution ---


def test_execute_moves_renames_file_and_updates_db(db, tmp_dir):
    src = tmp_dir / "source.jpg"
    src.write_bytes(b"hello")
    img_id = db.insert_image(
        normalize_path(src), "source.jpg", 5,
        exif_date="2022:04:01 09:15:00",
    )
    db.conn.commit()

    dest_root = tmp_dir / "archive"
    decisions, _ = plan_moves(db, dest_root)
    assert len(decisions) == 1

    moved, errors = execute_moves(db, decisions, Console())
    assert moved == 1
    assert errors == 0

    # File actually moved
    expected = dest_root / "2022" / "2022-04-01_091500.jpg"
    assert expected.exists()
    assert not src.exists()
    assert expected.read_bytes() == b"hello"

    # DB path updated to normalized form of new location
    row = db.conn.execute("SELECT path FROM images WHERE id = ?", (img_id,)).fetchone()
    assert row["path"] == normalize_path(expected)


def test_execute_moves_records_error_when_source_missing(db, tmp_dir):
    """A missing source file produces an error, leaves the DB row untouched."""
    ghost = tmp_dir / "ghost.jpg"  # never created
    original_path = normalize_path(ghost)
    img_id = db.insert_image(
        original_path, "ghost.jpg", 1,
        exif_date="2020:01:01 00:00:00",
    )
    db.conn.commit()

    dest_root = tmp_dir / "archive"
    decisions, _ = plan_moves(db, dest_root)
    moved, errors = execute_moves(db, decisions, Console())

    assert moved == 0
    assert errors == 1
    row = db.conn.execute("SELECT path FROM images WHERE id = ?", (img_id,)).fetchone()
    assert row["path"] == original_path


def test_execute_moves_is_idempotent(db, tmp_dir):
    """Running organize twice on the same DB produces no moves the second time."""
    src = tmp_dir / "twice.jpg"
    src.write_bytes(b"x")
    db.insert_image(
        normalize_path(src), "twice.jpg", 1,
        exif_date="2023:02:02 02:02:02",
    )
    db.conn.commit()

    dest_root = tmp_dir / "archive"

    decisions, _ = plan_moves(db, dest_root)
    moved, errors = execute_moves(db, decisions, Console())
    assert moved == 1 and errors == 0

    # Second pass: row's path now matches the computed destination, so no work.
    decisions2, skipped2 = plan_moves(db, dest_root)
    assert decisions2 == []
    assert skipped2 == 0


# --- Exclude-source filtering ---


def test_plan_moves_excludes_given_prefixes(db, tmp_dir):
    """exclude_sources=[prefix] drops every row whose path starts with prefix."""
    keep = tmp_dir / "keepers"
    drop = tmp_dir / "dropme"
    keep.mkdir()
    drop.mkdir()

    for folder, name in [(keep, "a.jpg"), (keep, "b.jpg"), (drop, "c.jpg"), (drop, "d.jpg")]:
        p = folder / name
        p.write_bytes(b"x")
        db.insert_image(
            normalize_path(p), name, 1,
            exif_date="2020:01:01 00:00:00",
        )
    db.conn.commit()

    decisions, _ = plan_moves(
        db, tmp_dir / "archive",
        exclude_sources=[str(drop)],
    )

    assert len(decisions) == 2
    srcs = {Path(d.src_path).name for d in decisions}
    assert srcs == {"a.jpg", "b.jpg"}


def test_plan_moves_exclude_respects_directory_boundary(db, tmp_dir):
    """--exclude-source foo must not also match foo_bar/... (prefix sibling)."""
    foo = tmp_dir / "foo"
    foobar = tmp_dir / "foo_bar"
    foo.mkdir()
    foobar.mkdir()

    in_foo = foo / "a.jpg"
    in_foo.write_bytes(b"x")
    in_foobar = foobar / "b.jpg"
    in_foobar.write_bytes(b"x")

    db.insert_image(normalize_path(in_foo), "a.jpg", 1, exif_date="2020:01:01 00:00:00")
    db.insert_image(normalize_path(in_foobar), "b.jpg", 1, exif_date="2020:01:01 00:00:00")
    db.conn.commit()

    decisions, _ = plan_moves(
        db, tmp_dir / "archive",
        exclude_sources=[str(foo)],
    )

    # Only the foo/ row is excluded; foo_bar/ survives.
    assert len(decisions) == 1
    assert Path(decisions[0].src_path).name == "b.jpg"


def test_plan_moves_empty_exclude_list_is_noop(db, tmp_dir):
    src = tmp_dir / "a.jpg"
    src.write_bytes(b"x")
    db.insert_image(normalize_path(src), "a.jpg", 1, exif_date="2020:01:01 00:00:00")
    db.conn.commit()

    decisions, _ = plan_moves(db, tmp_dir / "archive", exclude_sources=[])
    assert len(decisions) == 1


# --- Keyboard interrupt handling ---


def test_execute_moves_commits_pending_batch_on_interrupt(db, tmp_dir, monkeypatch):
    """Ctrl+C mid-loop must still persist updates for already-moved files.

    Regression guard for the batch-loss bug: prior to the try/finally wrap,
    interrupts would roll back up to 499 uncommitted path updates, leaving
    files moved on disk but DB pointing at their old locations.
    """
    import pytest

    from dedup import organizer

    # Three sources; we'll interrupt on the 3rd shutil.move.
    srcs = []
    for i in range(3):
        p = tmp_dir / f"src{i}.jpg"
        p.write_bytes(b"x")
        srcs.append(p)
        db.insert_image(
            normalize_path(p), p.name, 1,
            exif_date=f"2020:01:0{i + 1} 00:00:00",
        )
    db.conn.commit()

    dest_root = tmp_dir / "archive"
    decisions, _ = plan_moves(db, dest_root)
    assert len(decisions) == 3

    original_move = organizer.shutil.move
    call_count = {"n": 0}

    def flaky_move(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise KeyboardInterrupt()
        return original_move(*args, **kwargs)

    monkeypatch.setattr(organizer.shutil, "move", flaky_move)

    with pytest.raises(KeyboardInterrupt):
        execute_moves(db, decisions, Console())

    # After interrupt + finally commit, the first two rows' paths should
    # reflect the new archive location; the third should still be the source.
    rows = db.conn.execute(
        "SELECT id, path FROM images ORDER BY id"
    ).fetchall()
    assert "archive/2020" in rows[0]["path"]
    assert "archive/2020" in rows[1]["path"]
    assert "archive/2020" not in rows[2]["path"]
