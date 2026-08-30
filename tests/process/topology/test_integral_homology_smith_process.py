"""Killability tests for the integral-homology Smith worker boundary."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.math.matrices.certified_snf.operations import matrix_multiply
from jacobian.math.topology.chain_complexes import _smith_process
from jacobian.process import bounded_process_cancellation


def _sleeping_worker(path: Path) -> Path:
    path.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    return path


def _run_smith(deadline: float) -> _smith_process.SmithProcessResult:
    return _smith_process.smith_reduce_in_worker(
        [[2, 2], [2, 2]],
        rows=2,
        columns=2,
        deadline=deadline,
        left_bits=64,
        right_bits=64,
        diagonal_bits=64,
        left_inverse_bits=128,
        right_inverse_bits=128,
    )


def test_cancellation_kills_the_smith_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        _smith_process,
        "_SMITH_WORKER",
        _sleeping_worker(tmp_path / "sleeping_smith.py"),
    )
    cancellation = threading.Event()
    timer = threading.Timer(0.2, cancellation.set)
    started = time.monotonic()
    timer.start()
    try:
        with (
            bounded_process_cancellation(cancellation),
            pytest.raises(OperationExecutionCancelledError, match="cancelled"),
        ):
            _run_smith(started + 20)
    finally:
        timer.cancel()

    assert time.monotonic() - started < 3


def test_expired_deadline_prevents_smith_worker_launch() -> None:
    with pytest.raises(OperationExecutionTimeoutError, match="before Smith worker"):
        _run_smith(time.monotonic() - 1)


def test_deadline_kills_an_active_smith_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        _smith_process,
        "_SMITH_WORKER",
        _sleeping_worker(tmp_path / "sleeping_smith.py"),
    )
    started = time.monotonic()

    with pytest.raises(OperationExecutionTimeoutError, match="during Smith reduction"):
        _run_smith(started + 0.2)

    assert time.monotonic() - started < 3


def test_worker_setup_consumes_the_inherited_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    @contextmanager
    def delayed_directory(*args: Any, **kwargs: Any) -> Iterator[str]:
        del args, kwargs
        time.sleep(0.2)
        yield str(tmp_path)

    monkeypatch.setattr(_smith_process, "TemporaryDirectory", delayed_directory)

    with pytest.raises(OperationExecutionTimeoutError, match="before launching"):
        _run_smith(time.monotonic() + 0.1)


def test_worker_projection_is_bound_to_the_admitted_source() -> None:
    result = _run_smith(time.monotonic() + 20)

    assert (
        matrix_multiply(
            matrix_multiply(result.reduction.left, result.reduction.source),
            result.reduction.right,
        )
        == result.reduction.diagonal
    )
    assert matrix_multiply(result.left_inverse, result.reduction.left) == [
        [1, 0],
        [0, 1],
    ]


def test_wrong_worker_digest_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response: dict[str, Any] = {
        "request_digest": "0" * 64,
        "diagonal": [[2, 0], [0, 0]],
        "left": [[1, 0], [-1, 1]],
        "right": [[1, -1], [0, 1]],
        "rank": 1,
        "invariant_factors": [2],
        "left_determinant": 1,
        "right_determinant": 1,
        "left_inverse": [[1, 0], [1, 1]],
        "right_inverse": [[1, 1], [0, 1]],
    }
    worker = tmp_path / "wrong_digest.py"
    worker.write_text(
        "import json, sys\nsys.stdin.buffer.read()\n"
        f"json.dump({response!r}, sys.stdout, separators=(',', ':'))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_smith_process, "_SMITH_WORKER", worker)

    with pytest.raises(RuntimeError, match="malformed output"):
        _run_smith(time.monotonic() + 20)
