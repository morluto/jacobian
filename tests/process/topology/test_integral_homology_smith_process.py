"""Killability tests for the integral-homology Smith worker boundary."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.canonical import encode_strict_json, loads_strict_json
from jacobian.math.matrices.certified_snf.operations import (
    identity_matrix,
    matrix_determinant,
    matrix_multiply,
)
from jacobian.math.topology.chain_complexes import _smith_process, _smith_worker
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


def _run_singleton(value: int) -> _smith_process.SmithProcessResult:
    bits = value.bit_length() + 1
    return _smith_process.smith_reduce_in_worker(
        [[value]],
        rows=1,
        columns=1,
        deadline=time.monotonic() + 20,
        left_bits=bits,
        right_bits=bits,
        diagonal_bits=bits,
        left_inverse_bits=bits,
        right_inverse_bits=bits,
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


def test_launch_race_reports_typed_timeout_before_process_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((9.0, 10.0))
    launched = False

    def forbidden_launch(*args: Any, **kwargs: Any) -> Any:
        nonlocal launched
        launched = True
        raise AssertionError("expired allowance reached the process supervisor")

    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr("jacobian.process.run_bounded_process", forbidden_launch)

    with pytest.raises(OperationExecutionTimeoutError, match="before launching"):
        _run_smith(10.0)
    assert not launched


def test_worker_projection_is_bound_to_the_admitted_source() -> None:
    result = _run_smith(time.monotonic() + 20)

    assert (
        matrix_multiply(
            matrix_multiply(result.reduction.left, result.reduction.source),
            result.reduction.right,
        )
        == result.reduction.diagonal
    )
    unit = identity_matrix(2)
    assert matrix_multiply(result.left_inverse, result.reduction.left) == unit
    assert matrix_multiply(result.reduction.left, result.left_inverse) == unit
    assert matrix_multiply(result.right_inverse, result.reduction.right) == unit
    assert matrix_multiply(result.reduction.right, result.right_inverse) == unit
    assert (
        matrix_determinant(result.reduction.left) == result.reduction.left_determinant
    )
    assert (
        matrix_determinant(result.reduction.right) == result.reduction.right_determinant
    )


def test_worker_round_trips_integer_beyond_python_decimal_conversion_limit() -> None:
    value = 10**4_999 + 1

    result = _run_singleton(value)

    assert result.reduction.source == [[value]]
    assert result.reduction.diagonal == [[value]]
    assert result.reduction.invariant_factors == (value,)


def test_worker_encoder_uses_the_admitted_projection_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted_limit: int | None = None

    def inspect_launch(
        *args: Any, input_bytes: bytes, stdout_limit: int, **kwargs: Any
    ) -> Any:
        nonlocal admitted_limit
        del args, kwargs
        payload = loads_strict_json(input_bytes)
        admitted_limit = payload["output_limit"]
        assert admitted_limit == stdout_limit
        raise OSError("stop after inspecting the worker protocol")

    monkeypatch.setattr("jacobian.process.run_bounded_process", inspect_launch)

    with pytest.raises(RuntimeError, match="could not start"):
        _run_smith(time.monotonic() + 20)

    assert admitted_limit is not None
    assert admitted_limit == _smith_process._stdout_limit(
        rows=2,
        columns=2,
        left_bits=64,
        right_bits=64,
        diagonal_bits=64,
        left_inverse_bits=128,
        right_inverse_bits=128,
    )


def test_worker_applies_the_parent_projection_limit_to_canonical_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_limit = 12 * 1024 * 1024
    input_bytes = encode_strict_json(
        {
            "matrix": [["1"]],
            "row_count": 1,
            "column_count": 1,
            "output_limit": output_limit,
        }
    )
    stdout = BytesIO()
    observed_limit: int | None = None
    real_encode = encode_strict_json

    def inspect_encode(value: Any, *, limits: Any = None) -> bytes:
        nonlocal observed_limit
        observed_limit = limits.max_output_bytes
        return real_encode(value, limits=limits)

    projection = {
        "diagonal": [[1]],
        "left": [[1]],
        "right": [[1]],
        "rank": 1,
        "invariant_factors": (1,),
        "left_determinant": 1,
        "right_determinant": 1,
        "left_inverse": [[1]],
        "right_inverse": [[1]],
    }
    monkeypatch.setattr(
        _smith_worker, "_smith_projection", lambda *args, **kwargs: projection
    )
    monkeypatch.setattr(_smith_worker, "encode_strict_json", inspect_encode)
    monkeypatch.setattr(
        _smith_worker.sys, "stdin", SimpleNamespace(buffer=BytesIO(input_bytes))
    )
    monkeypatch.setattr(_smith_worker.sys, "stdout", SimpleNamespace(buffer=stdout))

    assert _smith_worker.main() == 0
    assert observed_limit == output_limit


def test_worker_decoder_rejects_noninteroperable_json_integer() -> None:
    with pytest.raises(ValueError, match="not interoperable"):
        _smith_process._strict_int(1 << 53)


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
