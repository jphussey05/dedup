"""Tests for perceptual hashing."""

from dedup.comparator import find_perceptual_duplicates
from dedup.perceptual import compute_perceptual_hashes
from dedup.scanner import scan_directory


def test_perceptual_finds_similar(db, sample_images):
    """Resized and recompressed images should be perceptually similar to original."""
    scan_directory(db, str(sample_images["dir"]), {".jpg"}, recursive=True)
    compute_perceptual_hashes(db)

    stats = db.get_status()
    assert stats["phashed"] == 5

    # Find perceptual duplicates with a generous threshold
    find_perceptual_duplicates(db, threshold=15)
    stats = db.get_status()
    # Original, exact_copy, resized, recompressed should cluster together
    assert stats["perceptual_groups"] >= 1


def test_perceptual_hash_idempotent(db, sample_images):
    """Running perceptual hash twice doesn't duplicate work."""
    scan_directory(db, str(sample_images["dir"]), {".jpg"}, recursive=True)
    compute_perceptual_hashes(db)
    compute_perceptual_hashes(db)

    stats = db.get_status()
    assert stats["phashed"] == 5
