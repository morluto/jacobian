"""Keep optional-provider feasibility probes isolated from the core catalog."""

from __future__ import annotations

import runpy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from tests.composition.runtime.provider_spike_isolation import (
    assert_unavailable_spike_preserves_catalog,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RunSpike = Callable[..., Mapping[str, object]]
SpikeArguments = Callable[[Path], dict[str, Path]]


def _run_spike(name: str) -> RunSpike:
    module = runpy.run_path(
        str(
            PROJECT_ROOT
            / "benchmarks"
            / "datasets"
            / "provider-feasibility-v1"
            / name
            / "environment"
            / "spike.py"
        )
    )
    return cast(RunSpike, module["run_spike"])


def _cddlib_paths(root: Path) -> dict[str, Path]:
    return {
        "python_executable": root / "absent-pycddlib-python",
        "cddlib_source_archive": root / "absent-cddlib.tar.gz",
        "pycddlib_source_archive": root / "absent-pycddlib.tar.gz",
    }


def _cgal_paths(root: Path) -> dict[str, Path]:
    return {
        "executable": root / "absent-cgal-spike",
        "source_archive": root / "absent-CGAL-6.2.tar.xz",
    }


def _gudhi_paths(root: Path) -> dict[str, Path]:
    return {
        "python_executable": root / "absent-gudhi-python",
        "wheel": root / "absent-gudhi.whl",
        "source_archive": root / "absent-gudhi-source.tar.gz",
    }


def _nauty_paths(root: Path) -> dict[str, Path]:
    return {
        "geng": root / "absent-geng",
        "labelg": root / "absent-labelg",
        "source_archive": root / "absent-nauty.tar.gz",
    }


def _regina_paths(root: Path) -> dict[str, Path]:
    return {
        "python_executable": root / "absent-regina-python",
        "wheel": root / "absent-regina.whl",
        "source_archive": root / "absent-regina-source.tar.gz",
    }


SPIKES: tuple[tuple[str, SpikeArguments], ...] = (
    ("cddlib", _cddlib_paths),
    ("cgal", _cgal_paths),
    ("gudhi", _gudhi_paths),
    ("nauty", _nauty_paths),
    ("regina", _regina_paths),
)


def test_unavailable_provider_spikes_preserve_complete_runtime_catalog(
    fresh_complete_runtime: Any,
    tmp_path: Path,
) -> None:
    """Feasibility probes must not register or remove core operations."""

    for name, arguments in SPIKES:
        run_spike = _run_spike(name)
        assert_unavailable_spike_preserves_catalog(
            fresh_complete_runtime,
            lambda run_spike=run_spike, arguments=arguments, name=name: run_spike(
                **arguments(tmp_path / name)
            ),
        )
