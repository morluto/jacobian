"""Strict Harbor job dataset/task selection models.

These models are the closed, ``extra="forbid"`` validation front door for
Harbor job configuration files parsed by ``benchmark_contracts`` and
``observation_results``.
"""

from __future__ import annotations

from pydantic import StrictStr

from benchmarks.tooling.strict_boundaries import _StrictModel


class HarborJobDatasetEntry(_StrictModel):
    path: StrictStr
    task_names: list[StrictStr] | None = None


class HarborJobTaskEntry(_StrictModel):
    path: StrictStr


class HarborJobSelection(_StrictModel):
    """Exactly one of ``datasets`` or ``tasks``; validated structurally first."""

    datasets: list[HarborJobDatasetEntry] | None = None
    tasks: list[HarborJobTaskEntry] | None = None


__all__ = [
    "HarborJobDatasetEntry",
    "HarborJobSelection",
    "HarborJobTaskEntry",
]
