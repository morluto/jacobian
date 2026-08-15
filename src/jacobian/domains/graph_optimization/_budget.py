"""Shared wall-clock budget helper for graph optimization operations."""

from __future__ import annotations

import time


def remaining_ms(started: float, wall_seconds: int) -> int:
    """Return the remaining wall-clock budget in milliseconds."""

    return int((wall_seconds - (time.monotonic() - started)) * 1000)
