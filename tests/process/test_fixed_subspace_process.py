"""Boundary tests for the finite-field fixed-subspace worker protocol."""

from __future__ import annotations

from time import monotonic

import pytest

import jacobian.process as process
from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
)
from jacobian.math.finite_fields._fixed_subspace_process import (
    _fixed_subspace_worker_input,
    run_fixed_subspace_computation,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


class _CancelDuringMarshalling:
    def __init__(self) -> None:
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks == 3


def test_generator_marshalling_observes_request_cancellation() -> None:
    matrix = PrimeFieldMatrix(
        prime=101,
        entries=((1, 0), (0, 1)),
        columns=2,
    )
    cancellation = _CancelDuringMarshalling()

    with (
        request_cancellation(cancellation),
        pytest.raises(OperationExecutionCancelledError, match="generator marshalling"),
    ):
        _fixed_subspace_worker_input((matrix, matrix), matrix, deadline=float("inf"))

    assert cancellation.checks == 3


def test_fixed_subspace_rejects_an_unbound_worker_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = PrimeFieldMatrix(
        prime=3,
        entries=((1, 0), (0, 1)),
        columns=2,
    )

    def unbound_projection(
        *args: object, **kwargs: object
    ) -> process.BoundedProcessResult:
        return process.BoundedProcessResult(
            returncode=0,
            stdout=(
                b'{"source_digest":"bad","generators_invertible":true,"basis_rows":[]}'
            ),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", unbound_projection)

    with pytest.raises(RuntimeError, match="malformed basis"):
        run_fixed_subspace_computation((matrix,), matrix, deadline=monotonic() + 10)
