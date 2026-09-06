"""Process-boundary regressions for affine-torus FLINT execution."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from threading import Event, Timer
from time import monotonic

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancellation,
    request_execution,
)
from jacobian.math.geometry.affine_tori import (
    RationalAffineTorusMap,
    _flint_process,
    affine_torus_fixed_locus,
)
from jacobian.math.geometry.affine_tori import _bounds as affine_bounds


def _source() -> RationalAffineTorusMap:
    translation = Fraction(0)
    return RationalAffineTorusMap.model_validate(
        {
            "torus": {"dimension": 1},
            "linear_part": {
                "row_count": 1,
                "column_count": 1,
                "entries": [[3]],
            },
            "translation": {
                "torus": {"dimension": 1},
                "coordinates": [
                    {
                        "num": translation.numerator,
                        "den": translation.denominator,
                    }
                ],
            },
        }
    )


def _hanging_worker(tmp_path: Path) -> tuple[Path, Path]:
    marker = tmp_path / "worker-started"
    worker = tmp_path / "hanging_worker.py"
    worker.write_text(
        "from pathlib import Path\n"
        "from time import sleep\n"
        f"Path({str(marker)!r}).write_text('started', encoding='utf-8')\n"
        "sleep(60)\n",
        encoding="utf-8",
    )
    return worker, marker


def test_cancellation_kills_the_affine_torus_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, marker = _hanging_worker(tmp_path)
    monkeypatch.setattr(_flint_process, "_AFFINE_TORUS_WORKER", worker)
    cancellation = Event()
    timer = Timer(1.0, cancellation.set)
    started = monotonic()
    timer.start()
    try:
        with (
            request_execution(started),
            request_cancellation(cancellation),
            pytest.raises(OperationExecutionCancelledError),
        ):
            affine_torus_fixed_locus(_source())
    finally:
        timer.cancel()
        timer.join()

    assert marker.is_file()
    assert monotonic() - started < 5.0


def test_deadline_kills_the_affine_torus_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, marker = _hanging_worker(tmp_path)
    monkeypatch.setattr(_flint_process, "_AFFINE_TORUS_WORKER", worker)
    monkeypatch.setattr(affine_bounds, "AFFINE_TORUS_FIXED_LOCUS_WALL_SECONDS", 1.5)
    monkeypatch.setattr(_flint_process, "_PARENT_FINALIZATION_SECONDS", 0.1)
    started = monotonic()

    with (
        request_execution(started),
        pytest.raises(OperationExecutionTimeoutError, match="execution allowance"),
    ):
        affine_torus_fixed_locus(_source())

    assert marker.is_file()
    assert monotonic() - started < 5.0
