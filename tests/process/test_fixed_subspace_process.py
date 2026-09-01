"""Boundary tests for the finite-field fixed-subspace worker protocol."""

from __future__ import annotations

import json

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
)
from jacobian.math.finite_fields._fixed_subspace_process import (
    _generator_ranks_stdout_limit,
    _generator_worker_input,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


class _CancelDuringMarshalling:
    def __init__(self) -> None:
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks == 3


def test_generator_rank_stdout_limit_covers_maximum_admitted_projection() -> None:
    response = json.dumps({"ranks": [100] * 1_024}, separators=(",", ":")).encode()

    assert len(response) == 4_107
    assert _generator_ranks_stdout_limit(1_024, 100) == len(response)


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
        _generator_worker_input((matrix, matrix), deadline=float("inf"))

    assert cancellation.checks == 3
