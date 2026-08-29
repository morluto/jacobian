"""Structural parsing and deliberate certificate verification for exact LPs."""

from __future__ import annotations

import copy
from fractions import Fraction
from typing import cast

import pytest
from tests.integration.linear._support import linear_validation_error
from tests.support.rationals import rational_payload as q

from jacobian._exact import CanonicalRational
from jacobian.math.optimization._models import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
    StandardFormRationalLinearProgram,
)
from jacobian.math.optimization.operations import linear_program

pytestmark = pytest.mark.requires_backend("sympy")


def _canonical(
    numerator: int,
    denominator: int = 1,
) -> CanonicalRational:
    return CanonicalRational.model_validate(q(numerator, denominator))


# minimize y subject to x+y=3 and x=1; the unique optimum is (1,2) with
# value 2 and a dual optimum (1,-1) that requires a free dual variable.
BOUND_PROGRAM: dict[str, object] = {
    "variables": ["x", "y"],
    "objective": [q(0), q(1)],
    "coefficients": [[q(1), q(1)], [q(1), q(0)]],
    "rhs": [q(3), q(1)],
}
INFEASIBLE_PROGRAM: dict[str, object] = {
    "variables": ["x", "y"],
    "objective": [q(0), q(0)],
    "coefficients": [[q(1), q(1)], [q(1), q(0)]],
    "rhs": [q(1), q(2)],
}
UNBOUNDED_PROGRAM: dict[str, object] = {
    "variables": ["x", "y"],
    "objective": [q(-1), q(0)],
    "coefficients": [[q(1), q(-1)]],
    "rhs": [q(0)],
}


def _solve(program: dict[str, object]) -> RationalLinearProgramResult:
    request = RationalLinearProgramRequest.model_validate({"program": program})
    return linear_program(request.program)


def _fractions(
    values: tuple[CanonicalRational, ...] | None,
) -> list[Fraction]:
    assert values is not None
    return [value.as_fraction() for value in values]


def _dot(left: list[Fraction], right: list[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def test_optimal_outcome_retains_source_and_replays_strong_duality() -> None:
    request = RationalLinearProgramRequest.model_validate({"program": BOUND_PROGRAM})
    result = linear_program(request.program)

    assert result.status == "OPTIMAL"
    assert result.program == request.program
    x = _fractions(result.primal_candidate)
    y = _fractions(result.dual_candidate)
    c = [Fraction(0), Fraction(1)]
    rows = [[Fraction(1), Fraction(1)], [Fraction(1), Fraction(0)]]
    b = [Fraction(3), Fraction(1)]
    assert all(value >= 0 for value in x)
    assert [_dot(row, x) for row in rows] == b
    assert [
        _dot(list(column), y) <= cj
        for column, cj in zip(zip(*rows, strict=True), c, strict=True)
    ]
    assert _dot(c, x) == _dot(b, y) == Fraction(2)
    assert _fractions(result.primal_residuals) == [Fraction(0), Fraction(0)]
    assert _fractions(result.dual_slacks) == [Fraction(0), Fraction(0)]


def test_infeasible_outcome_carries_a_replayable_farkas_candidate() -> None:
    result = _solve(INFEASIBLE_PROGRAM)

    assert result.status == "INFEASIBLE"
    assert result.primal_candidate is None
    witness = _fractions(result.farkas_candidate)
    rows = [[Fraction(1), Fraction(1)], [Fraction(1), Fraction(0)]]
    b = [Fraction(1), Fraction(2)]
    # Public sign convention: A^T y >= 0 and b^T y < 0.
    columns = [list(column) for column in zip(*rows, strict=True)]
    assert all(_dot(column, witness) >= 0 for column in columns)
    assert _dot(b, witness) < 0


def test_unbounded_outcome_carries_a_point_and_an_improving_ray() -> None:
    result = _solve(UNBOUNDED_PROGRAM)

    assert result.status == "UNBOUNDED"
    point = _fractions(result.primal_candidate)
    ray = _fractions(result.recession_direction)
    rows = [[Fraction(1), Fraction(-1)]]
    c = [Fraction(-1), Fraction(0)]
    assert all(value >= 0 for value in point)
    assert _dot(rows[0], point) == 0
    assert all(value >= 0 for value in ray)
    assert _dot(rows[0], ray) == 0
    assert _dot(c, ray) < 0


@pytest.mark.parametrize(
    ("program", "candidate"),
    [
        (
            {  # redundant rows: x=1 and 2x=2
                "variables": ["x"],
                "objective": [q(1)],
                "coefficients": [[q(1)], [q(2)]],
                "rhs": [q(1), q(2)],
            },
            None,
        ),
        (
            {  # degenerate zero objective with many optima
                "variables": ["x", "y"],
                "objective": [q(0), q(0)],
                "coefficients": [[q(1), q(1)]],
                "rhs": [q(1)],
            },
            None,
        ),
        (
            {  # exact fractional optimum at a rational boundary
                "variables": ["x", "y"],
                "objective": [q(-1), q(-1)],
                "coefficients": [[q(1), q(1)]],
                "rhs": [q(3, 2)],
            },
            None,
        ),
    ],
)
def test_admitted_edge_programs_stay_exact_and_bound_to_source(
    program: dict[str, object],
    candidate: list[Fraction] | None,
) -> None:
    request = RationalLinearProgramRequest.model_validate({"program": program})
    result = linear_program(request.program)

    assert result.status == "OPTIMAL"
    assert result.program == request.program
    if candidate is not None:
        assert _fractions(result.primal_candidate) == candidate


def test_empty_row_program_is_the_admitted_unconstrained_orthant() -> None:
    zero_row = copy.deepcopy(BOUND_PROGRAM)
    zero_row["coefficients"] = []
    zero_row["rhs"] = []
    program = StandardFormRationalLinearProgram.model_validate(zero_row)

    assert program.coefficients == ()
    assert program.rhs == ()


def test_fully_authored_optimal_payload_requires_structural_shape() -> None:
    program = StandardFormRationalLinearProgram.model_validate(BOUND_PROGRAM)
    forged = RationalLinearProgramResult(
        status="OPTIMAL",
        program=program,
        primal_candidate=(_canonical(999), _canonical(0)),
        primal_objective=_canonical(-123),
        primal_residuals=(_canonical(77), _canonical(77)),
        dual_candidate=(_canonical(456), _canonical(456)),
        dual_objective=_canonical(-123),
        dual_slacks=(_canonical(-88), _canonical(88)),
    )
    assert forged.status == "OPTIMAL"
    with linear_validation_error():
        RationalLinearProgramResult.model_validate(
            {
                "status": "OPTIMAL",
                "primal_candidate": [q(1), q(1)],
            }
        )


def _bound_result() -> tuple[RationalLinearProgramResult, dict[str, object]]:
    result = _solve(BOUND_PROGRAM)
    return result, result.model_dump(mode="json")


def _program_dump(dumped: dict[str, object]) -> dict[str, object]:
    program = dumped.get("program")
    assert isinstance(program, dict)
    return cast(dict[str, object], program)


def test_feasible_point_without_dual_remains_only_primal_feasible() -> None:
    program = StandardFormRationalLinearProgram.model_validate(BOUND_PROGRAM)
    feasible = RationalLinearProgramResult(
        status="PRIMAL_FEASIBLE",
        program=program,
        primal_candidate=(_canonical(1), _canonical(2)),
        primal_objective=_canonical(2),
        primal_residuals=(_canonical(0), _canonical(0)),
    )
    assert feasible.status == "PRIMAL_FEASIBLE"
    assert feasible.dual_candidate is None
    assert feasible.dual_objective is None
    with linear_validation_error():
        RationalLinearProgramResult.model_validate(
            {
                "status": "PRIMAL_FEASIBLE",
                "program": program,
                "primal_candidate": [q(1), q(2)],
                "primal_objective": q(2),
                "primal_residuals": [q(0), q(0)],
                "dual_candidate": [q(1), q(-1)],
            }
        )


def test_valid_farkas_certificate_round_trips_through_serialization() -> None:
    result = _solve(INFEASIBLE_PROGRAM)

    restored = RationalLinearProgramResult.model_validate(
        result.model_dump(mode="json")
    )
    assert restored == result
    assert restored.program.variables == ("x", "y")


def test_unknown_outcome_carries_no_mathematical_claim() -> None:
    program = StandardFormRationalLinearProgram.model_validate(BOUND_PROGRAM)
    unknown = RationalLinearProgramResult(status="UNKNOWN", program=program)

    assert unknown.status == "UNKNOWN"
    assert unknown.primal_candidate is None
    with linear_validation_error():
        RationalLinearProgramResult.model_validate(
            {
                "status": "UNKNOWN",
                "program": program,
                "primal_candidate": [q(1), q(1)],
            }
        )
    with linear_validation_error():
        RationalLinearProgramResult(status="UNBOUNDED", program=program)


def test_serialization_round_trip_preserves_binding_and_replay() -> None:
    for solve_program in (BOUND_PROGRAM, INFEASIBLE_PROGRAM, UNBOUNDED_PROGRAM):
        result = _solve(solve_program)
        dumped = result.model_dump(mode="json")
        restored = RationalLinearProgramResult.model_validate(dumped)

        assert restored == result
        assert restored.program.model_dump(mode="json") == dumped["program"]
