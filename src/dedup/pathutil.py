"""Cross-platform path normalization for local and NAS paths."""

from __future__ import annotations

import sys
from pathlib import Path


def normalize_path(p: str | Path) -> str:
    """Normalize a path for consistent storage and comparison.

    Handles mapped drives (Z:\\Photos) and UNC paths (\\\\NAS\\share).
    Stores with forward slashes, lowercased on Windows (NTFS is case-insensitive).
    """
    path = Path(p)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()

    result = str(resolved).replace("\\", "/")

    if sys.platform == "win32":
        result = result.lower()

    return result


def is_unc_path(p: str | Path) -> bool:
    """Check if a path is a UNC network path."""
    s = str(p)
    return s.startswith("//") or s.startswith("\\\\")


def get_read_chunk_size(p: str | Path) -> int:
    """Return optimal read chunk size based on path type.

    UNC/network paths benefit from larger chunks to reduce round-trips.
    """
    if is_unc_path(p):
        return 65536  # 64KB for network
    return 8192  # 8KB for local
