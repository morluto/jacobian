"""Tests for typed linear-system solve outcomes (#1940)."""

from __future__ import annotations

import pytest

from jacobian.math.matrices._operation_models import (
    RationalLinearSolveRequest,
)
from jacobian.math.matrices._operations import compute_rational_linear_solve


def _matrix(entries: list[list[str]]) -> list[list[dict]]:
    return [[{"num": e, "den": "1"} for e in row] for row in entries]


def _rhs(*values: str) -> list[dict]:
    return [{"num": v, "den": "1"} for v in values]


def test_unique_solution() -> None:
    """A nonsingular system returns a UNIQUE outcome with a solution."""
    request = RationalLinearSolveRequest.model_validate({
        "matrix": {"entries": _matrix([["1", "0"], ["0", "1"]])},
        "rhs": _rhs("2", "3"),
    })
    result = compute_rational_linear_solve(request)
    assert result.outcome == "UNIQUE"
    assert result.solution is not None
    assert len(result.solution) == 2


def test_inconsistent_system_returns_typed_outcome() -> None:
    """An inconsistent system returns INCONSISTENT, not a ValueError."""
    request = RationalLinearSolveRequest.model_validate({
        "matrix": {"entries": _matrix([["1", "1"], ["1", "1"]])},
        "rhs": _rhs("0", "1"),
    })
    result = compute_rational_linear_solve(request)
    assert result.outcome == "INCONSISTENT"
    assert result.solution is None


def test_non_unique_system_returns_typed_outcome() -> None:
    """A non-unique system returns NON_UNIQUE, not a ValueError."""
    request = RationalLinearSolveRequest.model_validate({
        "matrix": {"entries": _matrix([["1", "1"], ["1", "1"]])},
        "rhs": _rhs("1", "1"),
    })
    result = compute_rational_linear_solve(request)
    assert result.outcome == "NON_UNIQUE"
    assert result.solution is None


def test_singular_inverse_rejected() -> None:
    """Singular matrix inverse is rejected at the request boundary."""
    from jacobian.math.matrices._operation_models import (
        NonsingularIntegerMatrixRequest,
    )

    with pytest.raises(Exception, match="singular"):
        NonsingularIntegerMatrixRequest.model_validate({
            "matrix": {"entries": [["1", "2"], ["2", "4"]]}
        })
