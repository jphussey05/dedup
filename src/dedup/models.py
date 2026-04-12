"""Data models for dedup."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageRecord:
    id: int | None
    path: str
    filename: str
    file_size: int
    width: int | None = None
    height: int | None = None
    format: str | None = None
    exif_date: str | None = None
    sha256: str | None = None
    phash: str | None = None
    phash_int: int | None = None
    dhash: str | None = None
    dhash_int: int | None = None
    error: str | None = None


@dataclass
class DuplicateGroup:
    id: int | None
    match_type: str  # exact, perceptual, cnn
    confidence: float
    image_count: int


@dataclass
class DuplicatePair:
    id: int | None
    group_id: int
    image_id_1: int
    image_id_2: int
    similarity: float
