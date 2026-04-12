"""Tests for file discovery scanner."""

from dedup.scanner import scan_directory


def test_scan_finds_images(db, sample_images):
    """Scanner finds all image files in directory."""
    scan_directory(db, str(sample_images["dir"]), {".jpg", ".jpeg"}, recursive=True)

    stats = db.get_status()
    assert stats["total_images"] == 5


def test_scan_skips_existing(db, sample_images):
    """Running scan twice doesn't create duplicate rows."""
    scan_directory(db, str(sample_images["dir"]), {".jpg", ".jpeg"}, recursive=True)
    scan_directory(db, str(sample_images["dir"]), {".jpg", ".jpeg"}, recursive=True)

    stats = db.get_status()
    assert stats["total_images"] == 5


def test_scan_respects_format_filter(db, sample_images):
    """Scanner only includes files matching format filter."""
    scan_directory(db, str(sample_images["dir"]), {".png"}, recursive=True)

    stats = db.get_status()
    assert stats["total_images"] == 0
