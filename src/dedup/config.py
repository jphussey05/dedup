"""Configuration defaults and dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FORMATS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif",
    ".bmp", ".heic", ".heif", ".webp",
}


@dataclass
class Config:
    db_path: Path = field(default_factory=lambda: Path("dedup.db"))
    formats: set[str] = field(default_factory=lambda: DEFAULT_FORMATS.copy())
    recursive: bool = True
    batch_size: int = 1000
    phash_threshold: int = 10
    dhash_threshold: int = 10
    cnn_threshold: float = 0.92
    cnn_model: str = "clip-ViT-B-32"
    cnn_batch_size: int = 64
    cnn_device: str = "auto"
