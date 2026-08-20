"""Tests for typed linear-system solve outcomes (#1940)."""

from __future__ import annotations

import pytest
import sympy
from pydantic import ValidationError

from jacobian.math.matrices._operation_models import (
    RationalLinearSolveRequest,
)
from jacobian.math.matrices._operations import compute_rational_linear_solve


def _matrix(entries: list[list[str]]) -> list[list[dict]]:
    return [[{"num": e, "den": "1"} for e in row] for row in entries]


def _rhs(*values: str) -> list[dict]:
    return [{"num": v, "den": "1"} for v in values]


def _system_matrices(
    request: RationalLinearSolveRequest,
) -> tuple[sympy.Matrix, sympy.Matrix]:
    coefficients = sympy.Matrix(
        [
            [sympy.Rational(entry.num, entry.den) for entry in row]
            for row in request.matrix.entries
        ]
    )
    rhs = sympy.Matrix([sympy.Rational(entry.num, entry.den) for entry in request.rhs])
    return coefficients, rhs


def test_unique_solution() -> None:
    """A nonsingular system returns a UNIQUE outcome with a solution."""
    request = RationalLinearSolveRequest.model_validate(
        {
            "matrix": {"entries": _matrix([["1", "0"], ["0", "1"]])},
            "rhs": _rhs("2", "3"),
        }
    )
    result = compute_rational_linear_solve(request)
    assert result.outcome == "UNIQUE"
    assert result.solution is not None
    assert tuple((value.num, value.den) for value in result.solution) == (
        ("2", "1"),
        ("3", "1"),
    )
    assert result.convention == "LINEAR_SYSTEM_CLASSIFICATION_OVER_QQ"
    coefficients, rhs = _system_matrices(request)
    solution = sympy.Matrix(
        [sympy.Rational(value.num, value.den) for value in result.solution]
    )
    assert coefficients * solution == rhs
    assert coefficients.rank() == coefficients.row_join(rhs).rank() == coefficients.cols


def test_inconsistent_system_returns_typed_outcome() -> None:
    """An inconsistent system returns INCONSISTENT, not a ValueError."""
    request = RationalLinearSolveRequest.model_validate(
        {
            "matrix": {"entries": _matrix([["1", "1"], ["1", "1"]])},
            "rhs": _rhs("0", "1"),
        }
    )
    result = compute_rational_linear_solve(request)
    assert result.outcome == "INCONSISTENT"
    assert result.solution is None
    assert result.convention == "LINEAR_SYSTEM_CLASSIFICATION_OVER_QQ"
    coefficients, rhs = _system_matrices(request)
    assert coefficients.rank() < coefficients.row_join(rhs).rank()


def test_non_unique_system_returns_typed_outcome() -> None:
    """A non-unique system returns NON_UNIQUE, not a ValueError."""
    request = RationalLinearSolveRequest.model_validate(
        {
            "matrix": {"entries": _matrix([["1", "1"], ["1", "1"]])},
            "rhs": _rhs("1", "1"),
        }
    )
    result = compute_rational_linear_solve(request)
    assert result.outcome == "NON_UNIQUE"
    assert result.solution is None
    assert result.convention == "LINEAR_SYSTEM_CLASSIFICATION_OVER_QQ"
    coefficients, rhs = _system_matrices(request)
    assert coefficients.rank() == coefficients.row_join(rhs).rank() < coefficients.cols


def test_singular_inverse_rejected() -> None:
    """Singular matrix inverse is rejected at the request boundary."""
    from jacobian.math.matrices._operation_models import (
        NonsingularIntegerMatrixRequest,
    )

    with pytest.raises(ValidationError, match="singular"):
        NonsingularIntegerMatrixRequest.model_validate(
            {"matrix": {"entries": [["1", "2"], ["2", "4"]]}}
        )
