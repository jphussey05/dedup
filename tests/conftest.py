"""Shared fixtures for dedup tests."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from dedup.db import Database


@pytest.fixture
def tmp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def db(tmp_dir):
    """Create a temporary database."""
    db_path = tmp_dir / "test.db"
    database = Database(db_path)
    database.init_schema()
    yield database
    database.close()


@pytest.fixture
def sample_images(tmp_dir):
    """Create a set of test images for dedup testing.

    Returns a dict mapping name -> path for:
    - original: 100x100 red image
    - exact_copy: byte-identical copy
    - resized: 80x80 version (perceptual match)
    - recompressed: same image at JPEG quality 20 (perceptual match)
    - different: 100x100 blue image (no match)
    """
    img_dir = tmp_dir / "images"
    img_dir.mkdir()

    # Original: solid red 100x100
    original = Image.new("RGB", (100, 100), color=(255, 0, 0))
    original_path = img_dir / "original.jpg"
    original.save(str(original_path), "JPEG", quality=95)

    # Exact copy: byte-identical
    exact_path = img_dir / "exact_copy.jpg"
    shutil.copy2(str(original_path), str(exact_path))

    # Resized: 80x80 version
    resized = original.resize((80, 80))
    resized_path = img_dir / "resized.jpg"
    resized.save(str(resized_path), "JPEG", quality=95)

    # Recompressed: same image at low quality
    recompressed_path = img_dir / "recompressed.jpg"
    original.save(str(recompressed_path), "JPEG", quality=20)

    # Different: blue image
    different = Image.new("RGB", (100, 100), color=(0, 0, 255))
    different_path = img_dir / "different.jpg"
    different.save(str(different_path), "JPEG", quality=95)

    original.close()
    resized.close()
    different.close()

    return {
        "dir": img_dir,
        "original": original_path,
        "exact_copy": exact_path,
        "resized": resized_path,
        "recompressed": recompressed_path,
        "different": different_path,
    }
