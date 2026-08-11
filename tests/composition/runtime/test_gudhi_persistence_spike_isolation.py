from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from tests.composition.runtime.provider_spike_isolation import (
    assert_unavailable_spike_preserves_catalog,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPIKE = runpy.run_path(
    str(
        PROJECT_ROOT
        / "benchmarks"
        / "datasets"
        / "provider-feasibility-v1"
        / "gudhi"
        / "environment"
        / "spike.py"
    )
)
RunSpike = Callable[..., dict[str, Any]]
RUN_SPIKE = cast(RunSpike, SPIKE["run_spike"])


def test_absent_gudhi_does_not_change_complete_runtime_catalog(
    fresh_complete_runtime,
    tmp_path: Path,
) -> None:
    assert_unavailable_spike_preserves_catalog(
        fresh_complete_runtime,
        lambda: RUN_SPIKE(
            python_executable=tmp_path / "absent-gudhi-python",
            wheel=tmp_path / "absent-gudhi.whl",
            source_archive=tmp_path / "absent-gudhi-source.tar.gz",
        ),
    )
