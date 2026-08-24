"""Source binding and certificate replay for exact rational LP outcomes."""

from __future__ import annotations

import copy
from fractions import Fraction

import pytest
from pydantic import ValidationError
from tests.support.rationals import rational_payload as q

from jacobian.math.optimization._models import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
    StandardFormRationalLinearProgram,
)
from jacobian.math.optimization._tools import TOOLS as OPTIMIZATION_TOOLS

pytestmark = pytest.mark.requires_backend("sympy")

OPERATION = OPTIMIZATION_TOOLS[0]

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
    return OPERATION.run(
        RationalLinearProgramRequest.model_validate({"program": program})
    )


def _fractions(values: object) -> list[Fraction]:
    assert isinstance(values, tuple)
    return [value.as_fraction() for value in values]


def _dot(left: list[Fraction], right: list[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def test_optimal_outcome_retains_source_and_replays_strong_duality() -> None:
    request = RationalLinearProgramRequest.model_validate({"program": BOUND_PROGRAM})
    result = OPERATION.run(request)

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
    result = OPERATION.run(request)

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


def test_operation_version_tracks_the_source_bound_wire_shape() -> None:
    versions = {tool.operation_id: tool.version for tool in OPTIMIZATION_TOOLS}

    assert versions["optimization.linear.rational_optimum.compute"] == "2"


def test_fully_authored_optimal_payload_is_rejected() -> None:
    program = StandardFormRationalLinearProgram.model_validate(BOUND_PROGRAM)
    with pytest.raises(ValidationError):
        RationalLinearProgramResult(
            status="OPTIMAL",
            program=program,
            primal_candidate=(q(999), q(0)),
            primal_objective=q(-123),
            primal_residuals=(q(77), q(77)),
            dual_candidate=(q(456), q(456)),
            dual_objective=q(-123),
            dual_slacks=(q(-88), q(88)),
        )
    with pytest.raises(ValidationError, match="program"):
        RationalLinearProgramResult.model_validate(
            {
                "status": "OPTIMAL",
                "primal_candidate": [q(1), q(1)],
            }
        )


def _bound_result() -> tuple[RationalLinearProgramResult, dict[str, object]]:
    result = _solve(BOUND_PROGRAM)
    return result, result.model_dump(mode="json")


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("coefficients", [[q(1), q(1)], [q(2), q(0)]]),
        ("rhs", [q(2), q(3)]),
        ("objective", [q(0), q(2)]),
    ],
)
def test_mutating_one_source_entry_invalidates_a_valid_result(
    field: str,
    mutation: object,
) -> None:
    _, dumped = _bound_result()
    dumped["program"][field] = mutation  # type: ignore[literal-required]
    with pytest.raises(ValidationError):
        RationalLinearProgramResult.model_validate(dumped)


def test_variable_order_mutation_invalidates_a_valid_result() -> None:
    _, dumped = _bound_result()
    program = dumped["program"]
    assert isinstance(program, dict)
    program["variables"] = ["y", "x"]
    program["objective"] = [q(1), q(0)]
    program["coefficients"] = [[q(1), q(1)], [q(0), q(1)]]
    with pytest.raises(ValidationError):
        RationalLinearProgramResult.model_validate(dumped)


def test_mutated_primal_candidate_rejects_despite_matching_objective() -> None:
    _, dumped = _bound_result()
    # (2,2) keeps the submitted objective c^T x = 2 but violates x+y=3.
    dumped["primal_candidate"] = [q(2), q(2)]
    with pytest.raises(ValidationError):
        RationalLinearProgramResult.model_validate(dumped)


def test_mutated_dual_candidate_rejects_despite_matching_objective() -> None:
    _, dumped = _bound_result()
    # (2,-4) keeps the submitted objective b^T y = 2 but violates A^T y <= c.
    dumped["dual_candidate"] = [q(2), q(-4)]
    with pytest.raises(ValidationError):
        RationalLinearProgramResult.model_validate(dumped)


def test_nonzero_submitted_residuals_cannot_validate_as_optimal() -> None:
    _, dumped = _bound_result()
    dumped["primal_residuals"] = [q(0), q(7)]
    with pytest.raises(ValidationError, match="residuals"):
        RationalLinearProgramResult.model_validate(dumped)


def test_negative_submitted_slacks_cannot_validate_as_optimal() -> None:
    _, dumped = _bound_result()
    dumped["dual_slacks"] = [q(-1), q(0)]
    with pytest.raises(ValidationError, match="slacks"):
        RationalLinearProgramResult.model_validate(dumped)


def test_feasible_point_without_dual_remains_only_primal_feasible() -> None:
    program = StandardFormRationalLinearProgram.model_validate(BOUND_PROGRAM)
    feasible = RationalLinearProgramResult(
        status="PRIMAL_FEASIBLE",
        program=program,
        primal_candidate=(q(1), q(2)),
        primal_objective=q(2),
        primal_residuals=(q(0), q(0)),
    )
    assert feasible.status == "PRIMAL_FEASIBLE"
    assert feasible.dual_candidate is None
    assert feasible.dual_objective is None
    with pytest.raises(ValidationError, match="only an optimal"):
        RationalLinearProgramResult(
            status="PRIMAL_FEASIBLE",
            program=program,
            primal_candidate=(q(1), q(2)),
            primal_objective=q(2),
            primal_residuals=(q(0), q(0)),
            dual_candidate=(q(1), q(-1)),
        )


def test_corrupted_farkas_certificates_are_rejected() -> None:
    result = _solve(INFEASIBLE_PROGRAM)
    witness = _fractions(result.farkas_candidate)
    program = StandardFormRationalLinearProgram.model_validate(INFEASIBLE_PROGRAM)

    # Flipping every sign breaks A^T y >= 0 for this program.
    with pytest.raises(ValidationError, match="Farkas"):
        RationalLinearProgramResult(
            status="INFEASIBLE",
            program=program,
            farkas_candidate=tuple(
                q(-value.numerator, value.denominator) for value in witness
            ),
        )
    with pytest.raises(ValidationError, match="Farkas"):
        RationalLinearProgramResult(
            status="INFEASIBLE",
            program=program,
            farkas_candidate=(q(1),),
        )


def test_valid_farkas_certificate_round_trips_through_serialization() -> None:
    result = _solve(INFEASIBLE_PROGRAM)

    restored = RationalLinearProgramResult.model_validate(
        result.model_dump(mode="json")
    )
    assert restored == result
    assert restored.program.variables == ("x", "y")


def test_corrupted_unboundedness_pairs_are_rejected() -> None:
    result = _solve(UNBOUNDED_PROGRAM)
    program = StandardFormRationalLinearProgram.model_validate(UNBOUNDED_PROGRAM)
    ray = result.recession_direction

    def unbounded_with(
        direction: tuple[object, ...],
    ) -> RationalLinearProgramResult:
        return RationalLinearProgramResult(
            status="UNBOUNDED",
            program=program,
            primal_candidate=result.primal_candidate,
            primal_objective=result.primal_objective,
            primal_residuals=result.primal_residuals,
            recession_direction=direction,
        )

    # The zero ray satisfies Ad=0 but does not strictly improve c^T d.
    with pytest.raises(ValidationError, match="recession direction"):
        unbounded_with((q(0), q(0)))
    # (2,1) is nonnegative but violates A d = 0.
    with pytest.raises(ValidationError, match="Ad=0"):
        unbounded_with((q(2), q(1)))
    negated_ray = [q(-value.numerator, value.denominator) for value in _fractions(ray)]
    with pytest.raises(ValidationError, match="nonnegative"):
        unbounded_with(tuple(negated_ray))
    # An infeasible retained point cannot anchor an unbounded outcome.
    with pytest.raises(ValidationError, match="equalities"):
        RationalLinearProgramResult(
            status="UNBOUNDED",
            program=program,
            primal_candidate=(q(1), q(0)),
            primal_objective=q(-1),
            primal_residuals=(q(1),),
            recession_direction=ray,
        )


def test_dimension_mismatches_are_rejected_against_the_retained_source() -> None:
    result, dumped = _bound_result()
    program = result.program

    long_dump = copy.deepcopy(dumped)
    long_dump["primal_candidate"] = [q(1), q(1), q(1)]
    with pytest.raises(ValidationError, match="length must match the source"):
        RationalLinearProgramResult.model_validate(long_dump)
    with pytest.raises(ValidationError, match="length must match the source"):
        RationalLinearProgramResult(
            status="PRIMAL_FEASIBLE",
            program=program,
            primal_candidate=(q(1), q(1), q(1)),
            primal_objective=q(1),
            primal_residuals=(q(0), q(0)),
        )
    with pytest.raises(ValidationError, match="length must match the source"):
        RationalLinearProgramResult(
            status="OPTIMAL",
            program=program,
            primal_candidate=(q(1), q(2)),
            primal_objective=q(2),
            primal_residuals=(q(0), q(0)),
            dual_candidate=(q(1),),
            dual_objective=q(2),
            dual_slacks=(q(0), q(0)),
        )


def test_unknown_outcome_carries_no_mathematical_claim() -> None:
    program = StandardFormRationalLinearProgram.model_validate(BOUND_PROGRAM)
    unknown = RationalLinearProgramResult(status="UNKNOWN", program=program)

    assert unknown.status == "UNKNOWN"
    assert unknown.primal_candidate is None
    with pytest.raises(ValidationError, match="cannot carry primal data"):
        RationalLinearProgramResult(
            status="UNKNOWN",
            program=program,
            primal_candidate=(q(1), q(1)),
        )
    with pytest.raises(
        ValidationError, match="require exactly one feasible primal point"
    ):
        RationalLinearProgramResult(status="UNBOUNDED", program=program)


def test_serialization_round_trip_preserves_binding_and_replay() -> None:
    for solve_program in (BOUND_PROGRAM, INFEASIBLE_PROGRAM, UNBOUNDED_PROGRAM):
        result = _solve(solve_program)
        dumped = result.model_dump(mode="json")
        restored = RationalLinearProgramResult.model_validate(dumped)

        assert restored == result
        assert restored.program.model_dump(mode="json") == dumped["program"]
