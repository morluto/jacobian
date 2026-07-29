from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPIKE = runpy.run_path(str(PROJECT_ROOT / "benchmarks" / "cgal_delaunay_spike.py"))
RunSpike = Callable[..., dict[str, Any]]
RUN_SPIKE = cast(RunSpike, SPIKE["run_spike"])


def test_absent_cgal_spike_does_not_change_complete_runtime_catalog(
    fresh_complete_runtime,
    tmp_path: Path,
) -> None:
    before = fresh_complete_runtime.core.capabilities.catalog().model_dump(mode="json")

    report = RUN_SPIKE(
        executable=tmp_path / "absent-cgal-spike",
        source_archive=tmp_path / "absent-CGAL-6.2.tar.xz",
    )

    after = fresh_complete_runtime.core.capabilities.catalog().model_dump(mode="json")
    assert report["status"] == "UNAVAILABLE"
    assert report["capability_ids_registered"] == []
    assert after == before
