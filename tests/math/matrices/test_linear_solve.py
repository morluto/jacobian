"""Tests for typed linear-system solve outcomes (#1940)."""

from __future__ import annotations

import copy

import pytest
import sympy
from pydantic import ValidationError

from jacobian.math.matrices._operation_models import (
    RationalLinearSolveRequest,
    RationalLinearSolveResult,
)
from jacobian.math.matrices._operations import (
    compute_inverse,
    compute_rational_linear_solve,
    verify_rational_linear_solve_result,
)


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


def _mutable(dumped: dict) -> dict:
    """JSON round-trip so nested tuple payloads become mutable lists."""
    import json

    return json.loads(json.dumps(dumped))


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
        assert verify_rational_linear_solve_result(result)
        assert (
            RationalLinearSolveResult.model_validate_json(result.model_dump_json())
            == result
        )


def test_unique_result_rejects_forged_solution_mutations() -> None:
    """A mutated solution coordinate fails the A x = b replay."""

    request = RationalLinearSolveRequest.model_validate(
        {
            "matrix": {"entries": _matrix([["2", "0"], ["0", "3"]])},
            "rhs": _rhs("2", "3"),
        }
    )
    dumped = _mutable(compute_rational_linear_solve(request).model_dump())

    forged_coordinate = copy.deepcopy(dumped)
    forged_coordinate["solution"][1] = {"num": "4", "den": "1"}
    assert not verify_rational_linear_solve_result(
        RationalLinearSolveResult.model_validate(forged_coordinate)
    )

    forged_length = copy.deepcopy(dumped)
    forged_length["solution"] = [
        {"num": "1", "den": "1"},
        {"num": "1", "den": "1"},
        {"num": "1", "den": "1"},
    ]
    with pytest.raises(ValidationError):
        RationalLinearSolveResult.model_validate(forged_length)

    missing_solution = copy.deepcopy(dumped)
    missing_solution["solution"] = None
    with pytest.raises(ValidationError):
        RationalLinearSolveResult.model_validate(missing_solution)


def test_results_reject_outcome_and_source_mutations() -> None:
    """Outcome flips and source edits fail the classification replay."""

    inconsistent_request = RationalLinearSolveRequest.model_validate(
        {
            "matrix": {"entries": _matrix([["1", "1"], ["1", "1"]])},
            "rhs": _rhs("0", "1"),
        }
    )
    inconsistent = _mutable(
        compute_rational_linear_solve(inconsistent_request).model_dump()
    )

    flipped_non_unique = copy.deepcopy(inconsistent)
    flipped_non_unique["outcome"] = "NON_UNIQUE"
    assert not verify_rational_linear_solve_result(
        RationalLinearSolveResult.model_validate(flipped_non_unique)
    )

    flipped_unique = copy.deepcopy(inconsistent)
    flipped_unique["outcome"] = "UNIQUE"
    with pytest.raises(ValidationError):
        RationalLinearSolveResult.model_validate(flipped_unique)

    feasible_source = copy.deepcopy(inconsistent)
    feasible_source["rhs"] = _rhs("0", "0")
    assert not verify_rational_linear_solve_result(
        RationalLinearSolveResult.model_validate(feasible_source)
    )

    nonsingular_source = copy.deepcopy(inconsistent)
    nonsingular_source["matrix"]["entries"] = _matrix([["1", "0"], ["0", "1"]])
    assert not verify_rational_linear_solve_result(
        RationalLinearSolveResult.model_validate(nonsingular_source)
    )

    nonunique_request = RationalLinearSolveRequest.model_validate(
        {
            "matrix": {"entries": _matrix([["1", "1"], ["1", "1"]])},
            "rhs": _rhs("1", "1"),
        }
    )
    non_unique = _mutable(compute_rational_linear_solve(nonunique_request).model_dump())

    flipped_inconsistent = copy.deepcopy(non_unique)
    flipped_inconsistent["outcome"] = "INCONSISTENT"
    assert not verify_rational_linear_solve_result(
        RationalLinearSolveResult.model_validate(flipped_inconsistent)
    )

    singular_to_nonsingular = copy.deepcopy(non_unique)
    singular_to_nonsingular["matrix"]["entries"] = _matrix([["1", "0"], ["0", "1"]])
    singular_to_nonsingular["rhs"] = _rhs("1", "1")
    assert not verify_rational_linear_solve_result(
        RationalLinearSolveResult.model_validate(singular_to_nonsingular)
    )

    unique_request = RationalLinearSolveRequest.model_validate(
        {
            "matrix": {"entries": _matrix([["2", "0"], ["0", "3"]])},
            "rhs": _rhs("2", "3"),
        }
    )
    unique = _mutable(compute_rational_linear_solve(unique_request).model_dump())

    foreign_rhs = copy.deepcopy(unique)
    foreign_rhs["rhs"] = _rhs("2", "4")
    assert not verify_rational_linear_solve_result(
        RationalLinearSolveResult.model_validate(foreign_rhs)
    )

    # A singular source whose claimed solution still satisfies A x = b
    # exactly: only the nonsingularity replay can reject this forgery.
    singular_source = {
        "matrix": {"entries": _matrix([["1", "1"], ["2", "2"]])},
        "rhs": _rhs("2", "4"),
        "outcome": "UNIQUE",
        "solution": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
    }
    assert not verify_rational_linear_solve_result(
        RationalLinearSolveResult.model_validate(singular_source)
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


def test_result_reapplies_source_admission_without_replay() -> None:
    """A relayed source outside the work envelope is rejected structurally."""
    unadmitted = {
        "matrix": {
            "entries": [
                [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
                [{"num": "0", "den": "1"}, {"num": "9" * 257, "den": "1"}],
            ]
        },
        "rhs": _rhs("1", "1"),
        "outcome": "INCONSISTENT",
    }
    with pytest.raises(ValidationError):
        RationalLinearSolveResult.model_validate(unadmitted)

    non_square = dict(unadmitted)
    non_square["matrix"] = {"entries": [[{"num": "1", "den": "1"}]]}
    with pytest.raises(ValidationError):
        RationalLinearSolveResult.model_validate(non_square)
