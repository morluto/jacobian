"""Compatibility adapter for package-owned benchmark path decoding."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_plan.paths import normalize_paths, path_values  # noqa: E402

__all__ = ["normalize_paths", "path_values"]
