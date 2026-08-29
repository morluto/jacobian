"""Tests for typed linear-system solve outcomes (#1940)."""

from __future__ import annotations

import pytest
import sympy
from pydantic import ValidationError

from jacobian.math.matrices._operation_models import (
    RationalLinearSolveRequest,
    RationalLinearSolveResult,
)
from jacobian.math.matrices._tools import (
    compute_inverse,
    compute_rational_linear_solve,
)


def _matrix(entries: list[list[str]]) -> list[list[dict[str, str]]]:
    return [[{"num": e, "den": "1"} for e in row] for row in entries]


def _rhs(*values: str) -> list[dict[str, str]]:
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


def test_singular_inverse_rejected_by_the_exact_kernel() -> None:
    """Request parsing stays structural; the inverse kernel rejects singularity."""
    from jacobian.math.matrices._operation_models import (
        NonsingularIntegerMatrixRequest,
    )

    request = NonsingularIntegerMatrixRequest.model_validate(
        {"matrix": {"entries": [["1", "2"], ["2", "4"]]}}
    )
    with pytest.raises(ValueError, match="singular"):
        compute_inverse(request)


def test_results_retain_their_source_system() -> None:
    """Every outcome retains the exact coefficient matrix and right-hand side."""

    cases = (
        (_matrix([["1", "0"], ["0", "1"]]), _rhs("2", "3"), "UNIQUE"),
        (_matrix([["1", "1"], ["1", "1"]]), _rhs("0", "1"), "INCONSISTENT"),
        (_matrix([["1", "1"], ["1", "1"]]), _rhs("1", "1"), "NON_UNIQUE"),
    )
    for entries, rhs, outcome in cases:
        request = RationalLinearSolveRequest.model_validate(
            {"matrix": {"entries": entries}, "rhs": rhs}
        )
        result = compute_rational_linear_solve(request)
        assert result.outcome == outcome
        assert result.matrix == request.matrix
        assert result.rhs == request.rhs
        assert (
            RationalLinearSolveResult.model_validate_json(result.model_dump_json())
            == result
        )


def test_serialized_results_round_trip_through_the_wire_shape() -> None:
    """Producer output validates through a JSON round trip on every outcome."""

    requests = (
        RationalLinearSolveRequest.model_validate(
            {
                "matrix": {
                    "entries": [
                        [{"num": "1", "den": "2"}, {"num": "0", "den": "1"}],
                        [{"num": "0", "den": "1"}, {"num": "1", "den": "3"}],
                    ]
                },
                "rhs": [{"num": "1", "den": "2"}, {"num": "-5", "den": "3"}],
            }
        ),
        RationalLinearSolveRequest.model_validate(
            {
                "matrix": {
                    "entries": _matrix(
                        [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
                    )
                },
                "rhs": _rhs("1", "1", "1"),
            }
        ),
    )
    for request in requests:
        result = compute_rational_linear_solve(request)
        restored = RationalLinearSolveResult.model_validate(result.model_dump())
        assert restored == result
        assert restored.convention == "LINEAR_SYSTEM_CLASSIFICATION_OVER_QQ"


def test_result_rejects_source_shape_mismatch() -> None:
    non_square = {
        "matrix": {"entries": [[{"num": "1", "den": "1"}]]},
        "rhs": _rhs("1", "1"),
        "outcome": "INCONSISTENT",
    }
    with pytest.raises(ValidationError):
        RationalLinearSolveResult.model_validate(non_square)
