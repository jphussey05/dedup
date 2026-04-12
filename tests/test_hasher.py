"""Tests for SHA256 hashing."""

from dedup.comparator import find_exact_duplicates
from dedup.hasher import hash_images
from dedup.scanner import scan_directory


def test_hash_finds_exact_duplicates(db, sample_images):
    """Exact copies should be detected as duplicates."""
    scan_directory(db, str(sample_images["dir"]), {".jpg"}, recursive=True)
    hash_images(db)
    find_exact_duplicates(db)

    stats = db.get_status()
    assert stats["exact_groups"] >= 1


def test_hash_idempotent(db, sample_images):
    """Running hash twice produces same results."""
    scan_directory(db, str(sample_images["dir"]), {".jpg"}, recursive=True)
    hash_images(db)
    hash_images(db)

    stats = db.get_status()
    assert stats["hashed"] == 5
