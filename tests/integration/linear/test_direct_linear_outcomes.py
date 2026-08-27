from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from tests.integration.linear._support import linear_validation_error
from tests.support.rationals import rational_payload as q

from jacobian.math.matrices.rational_linear._models import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalSolutionFindRequest,
)
from jacobian.math.matrices.rational_linear._operations import (
    compute_rational_inconsistency,
    compute_rational_solution,
)
from jacobian.math.optimization._models import (
    RationalLinearProgramRequest,
    RationalLinearProgramResult,
    StandardFormRationalLinearProgram,
    _verify_rational_linear_program_result,
)
from jacobian.math.optimization._tools import TOOLS as OPTIMIZATION_TOOLS

pytestmark = pytest.mark.requires_backend("flint")


def _assert_rejected_by_verifier(result: RationalLinearProgramResult) -> None:
    with pytest.raises(ValueError):
        _verify_rational_linear_program_result(result)


def _system(rhs: list[dict[str, str]]) -> dict[str, object]:
    return {
        "variables": ["x"],
        "coefficients": {"entries": [[q(1)] for _ in rhs]},
        "rhs": rhs,
    }


def _run_linear_program(program: dict[str, object]) -> RationalLinearProgramResult:
    request = RationalLinearProgramRequest.model_validate({"program": program})
    result = OPTIMIZATION_TOOLS[0].run(request)
    assert isinstance(result, RationalLinearProgramResult)
    return result


def test_rational_linear_operations_return_mathematical_outcomes() -> None:
    consistent = _system([q(1)])
    inconsistent = _system([q(1), q(2)])

    solution = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": consistent})
    )
    no_solution = compute_rational_solution(
        LinearRationalSolutionFindRequest.model_validate({"system": inconsistent})
    )
    consistency = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest.model_validate({"system": consistent})
    )
    contradiction = compute_rational_inconsistency(
        LinearRationalInconsistencyFindRequest.model_validate({"system": inconsistent})
    )

    assert solution.status == "SOLUTION"
    assert solution.values is not None
    assert [v.model_dump(mode="json") for v in solution.values] == [q(1)]
    assert no_solution.status == "INCONSISTENT"
    assert consistency.status == "CONSISTENT"
    assert contradiction.status == "INCONSISTENT"
    assert contradiction.left_witness is not None
    assert contradiction.rhs_pairing is not None
    witness = tuple(value.as_fraction() for value in contradiction.left_witness)
    assert witness == (Fraction(-1), Fraction(1))
    assert sum(coordinate * Fraction(1) for coordinate in witness) == 0
    assert sum(
        coordinate * value
        for coordinate, value in zip(witness, (Fraction(1), Fraction(2)), strict=True)
    ) == Fraction(1)
    assert contradiction.rhs_pairing.model_dump(mode="json") == q(1)


def test_rational_linear_program_returns_a_source_bound_optimum() -> None:
    operation = OPTIMIZATION_TOOLS[0]
    result = operation.run(
        RationalLinearProgramRequest.model_validate(
            {
                "program": {
                    "variables": ["x"],
                    "objective": [q(1)],
                    "coefficients": [[q(1)]],
                    "rhs": [q(1)],
                }
            }
        )
    )

    assert result.status == "OPTIMAL"
    # Guard the public wire shape: every conclusion retains the canonical source,
    # while status-specific witnesses stay direct mathematical values.
    assert set(result.model_dump(mode="json")) == {
        "program",
        "status",
        "primal_candidate",
        "dual_candidate",
        "primal_objective",
        "dual_objective",
        "primal_residuals",
        "dual_slacks",
        "farkas_candidate",
        "recession_direction",
    }
    assert (
        result.program
        == RationalLinearProgramRequest.model_validate(
            {
                "program": {
                    "variables": ["x"],
                    "objective": [q(1)],
                    "coefficients": [[q(1)]],
                    "rhs": [q(1)],
                }
            }
        ).program
    )
    assert [v.model_dump(mode="json") for v in result.primal_candidate] == [q(1)]
    assert [v.model_dump(mode="json") for v in result.dual_candidate] == [q(1)]
    assert result.primal_objective.model_dump(mode="json") == q(1)
    assert result.dual_objective.model_dump(mode="json") == q(1)
    assert [v.model_dump(mode="json") for v in result.primal_residuals] == [q(0)]
    assert [v.model_dump(mode="json") for v in result.dual_slacks] == [q(0)]


def test_rational_linear_program_handles_multiple_equalities() -> None:
    operation = OPTIMIZATION_TOOLS[0]
    result = operation.run(
        RationalLinearProgramRequest.model_validate(
            {
                "program": {
                    "variables": ["x", "y"],
                    "objective": [q(1), q(1)],
                    "coefficients": [[q(1), q(0)], [q(0), q(1)]],
                    "rhs": [q(1), q(2)],
                }
            }
        )
    )

    assert result.status == "OPTIMAL"
    assert [v.model_dump(mode="json") for v in result.primal_candidate] == [q(1), q(2)]
    assert [v.model_dump(mode="json") for v in result.primal_residuals] == [q(0), q(0)]


def test_rational_linear_program_returns_a_replayable_farkas_witness() -> None:
    result = _run_linear_program(
        {
            "variables": ["x", "y"],
            "objective": [q(3), q(-2)],
            "coefficients": [
                [q(1), q(0)],
                [q(0), q(1)],
                [q(1), q(1)],
            ],
            "rhs": [q(0), q(0), q(1)],
        }
    )

    assert result.status == "INFEASIBLE"
    assert result.farkas_candidate is not None
    witness = tuple(value.as_fraction() for value in result.farkas_candidate)
    assert witness == (Fraction(1), Fraction(1), Fraction(-1))
    assert result.primal_candidate is None
    assert (
        RationalLinearProgramResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_zero_coefficient_nonzero_rhs_has_a_direct_farkas_witness() -> None:
    result = _run_linear_program(
        {
            "variables": ["x", "y"],
            "objective": [q(0), q(0)],
            "coefficients": [[q(0), q(0)], [q(1), q(-1)]],
            "rhs": [q(-3), q(0)],
        }
    )

    assert result.status == "INFEASIBLE"
    assert result.farkas_candidate is not None
    assert tuple(value.as_fraction() for value in result.farkas_candidate) == (
        Fraction(1),
        Fraction(0),
    )


def test_zero_equalities_are_pruned_from_farkas_backend_work() -> None:
    from jacobian.math.optimization._operations import _solve_farkas

    program = RationalLinearProgramRequest.model_validate(
        {
            "program": {
                "variables": ["x", "y"],
                "objective": [q(0), q(0)],
                "coefficients": [[q(1), q(0)], [q(1), q(0)]]
                + [[q(0), q(0)] for _ in range(62)],
                "rhs": [q(0), q(1)] + [q(0) for _ in range(62)],
            }
        }
    ).program

    active_rows, positive, negative, _ = _solve_farkas(program)
    assert active_rows == (0, 1)
    assert len(positive) == len(negative) == 2

    result = _run_linear_program(program.model_dump(mode="json"))
    assert result.status == "INFEASIBLE"
    assert result.farkas_candidate is not None
    assert len(result.farkas_candidate) == 64
    assert all(value.as_fraction() == 0 for value in result.farkas_candidate[2:])


def test_rational_linear_program_returns_a_feasible_point_and_recession_ray() -> None:
    result = _run_linear_program(
        {
            "variables": ["x", "y"],
            "objective": [q(-1), q(0)],
            "coefficients": [[q(1), q(-1)]],
            "rhs": [q(1)],
        }
    )

    assert result.status == "UNBOUNDED"
    assert result.primal_candidate is not None
    assert result.recession_direction is not None
    point = tuple(value.as_fraction() for value in result.primal_candidate)
    direction = tuple(value.as_fraction() for value in result.recession_direction)
    assert point[0] - point[1] == 1
    assert all(value >= 0 for value in point)
    assert direction[0] - direction[1] == 0
    assert all(value >= 0 for value in direction)
    assert -direction[0] < 0


@pytest.mark.parametrize(
    ("objective", "expected_status"),
    [
        ([q(1), q(2)], "OPTIMAL"),
        ([q(-1), q(2)], "UNBOUNDED"),
    ],
)
def test_rational_linear_program_has_an_explicit_zero_row_convention(
    objective: list[dict[str, str]],
    expected_status: str,
) -> None:
    result = _run_linear_program(
        {
            "variables": ["x", "y"],
            "objective": objective,
            "coefficients": [],
            "rhs": [],
        }
    )

    assert result.status == expected_status
    assert result.program.coefficients == ()
    assert result.program.rhs == ()
    assert result.primal_candidate is not None
    assert result.primal_residuals == ()


def test_rational_linear_program_handles_redundancy_degeneracy_and_rationals() -> None:
    result = _run_linear_program(
        {
            "variables": ["x", "y"],
            "objective": [q(1, 2), q(1, 2)],
            "coefficients": [
                [q(1, 3), q(1, 3)],
                [q(2, 3), q(2, 3)],
            ],
            "rhs": [q(1, 3), q(2, 3)],
        }
    )

    assert result.status == "OPTIMAL"
    assert result.primal_objective is not None
    assert result.dual_objective is not None
    assert result.primal_objective.as_fraction() == Fraction(1, 2)
    assert result.dual_objective.as_fraction() == Fraction(1, 2)
    assert result.primal_residuals is not None
    assert all(value.as_fraction() == 0 for value in result.primal_residuals)


def test_source_derived_result_height_exceeds_input_scalar_limit() -> None:
    scale = 10**127
    first_rhs = Fraction(scale + 6, scale + 2)
    result = _run_linear_program(
        {
            "variables": ["x", "y"],
            "objective": [q(0), q(0)],
            "coefficients": [
                [q(1, scale + 1), q(1, scale + 3)],
                [q(1, scale + 5), q(1, scale + 7)],
            ],
            "rhs": [q(first_rhs.numerator, first_rhs.denominator), q(1)],
        }
    )

    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None
    assert max(len(value.num.lstrip("-")) for value in result.primal_candidate) > 128
    assert (
        RationalLinearProgramResult.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )


def test_optimal_result_rejects_source_and_diagnostic_mutations() -> None:
    result = _run_linear_program(
        {
            "variables": ["x"],
            "objective": [q(1)],
            "coefficients": [[q(1)]],
            "rhs": [q(1)],
        }
    )
    payload = result.model_dump(mode="json")
    mutations: list[tuple[tuple[str, ...], object]] = [
        (("program", "rhs"), [q(2)]),
        (("primal_candidate",), [q(2)]),
        (("primal_objective",), q(2)),
        (("primal_residuals",), [q(1)]),
        (("dual_candidate",), [q(0)]),
        (("dual_objective",), q(0)),
        (("dual_slacks",), [q(1)]),
    ]

    for path, replacement in mutations:
        mutated = deepcopy(payload)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        _assert_rejected_by_verifier(
            RationalLinearProgramResult.model_validate(mutated)
        )


def test_optimal_result_rejects_matrix_objective_and_variable_order_mutations() -> None:
    result = _run_linear_program(
        {
            "variables": ["x", "y"],
            "objective": [q(1), q(3)],
            "coefficients": [[q(1), q(0)], [q(0), q(1)]],
            "rhs": [q(1), q(2)],
        }
    )
    payload = result.model_dump(mode="json")
    mutations = []

    coefficient = deepcopy(payload)
    coefficient["program"]["coefficients"][0][0] = q(2)
    mutations.append(coefficient)
    rhs = deepcopy(payload)
    rhs["program"]["rhs"][0] = q(2)
    mutations.append(rhs)
    objective = deepcopy(payload)
    objective["program"]["objective"][0] = q(2)
    mutations.append(objective)
    row = deepcopy(payload)
    row["program"]["coefficients"] = [[q(0), q(1)], [q(1), q(0)]]
    mutations.append(row)
    variable_order = deepcopy(payload)
    variable_order["program"]["variables"] = ["y", "x"]
    variable_order["program"]["objective"] = [q(3), q(1)]
    variable_order["program"]["coefficients"] = [
        [q(0), q(1)],
        [q(1), q(0)],
    ]
    mutations.append(variable_order)

    for mutation in mutations:
        _assert_rejected_by_verifier(
            RationalLinearProgramResult.model_validate(mutation)
        )


def test_missing_dual_evidence_remains_primal_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sympy.solvers import simplex

    def unavailable_dual(*_args: object, **_kwargs: object) -> None:
        raise simplex.InfeasibleLPError("dual certificate unavailable")

    monkeypatch.setattr(simplex, "lpmax", unavailable_dual)
    result = _run_linear_program(
        {
            "variables": ["x"],
            "objective": [q(1)],
            "coefficients": [[q(1)]],
            "rhs": [q(1)],
        }
    )

    assert result.status == "PRIMAL_FEASIBLE"
    assert result.primal_candidate is not None
    assert result.dual_candidate is None


def test_missing_negative_certificate_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sympy.solvers import simplex

    def unavailable_simplex(*_args: object, **_kwargs: object) -> None:
        raise simplex.InfeasibleLPError("certificate unavailable")

    monkeypatch.setattr(simplex, "lpmin", unavailable_simplex)
    result = _run_linear_program(
        {
            "variables": ["x"],
            "objective": [q(0)],
            "coefficients": [[q(1)], [q(1)]],
            "rhs": [q(0), q(1)],
        }
    )

    assert result.status == "UNKNOWN"
    assert result.farkas_candidate is None
    assert result.recession_direction is None


def test_negative_results_reject_bare_or_source_mutated_claims() -> None:
    infeasible = _run_linear_program(
        {
            "variables": ["x"],
            "objective": [q(0)],
            "coefficients": [[q(1)], [q(1)]],
            "rhs": [q(0), q(1)],
        }
    ).model_dump(mode="json")
    unbounded = _run_linear_program(
        {
            "variables": ["x", "y"],
            "objective": [q(-1), q(0)],
            "coefficients": [[q(1), q(-1)]],
            "rhs": [q(1)],
        }
    ).model_dump(mode="json")

    mutations = []
    bare_infeasible = deepcopy(infeasible)
    bare_infeasible["farkas_candidate"] = None
    mutations.append(bare_infeasible)
    zero_farkas = deepcopy(infeasible)
    zero_farkas["farkas_candidate"] = [q(0), q(0)]
    mutations.append(zero_farkas)
    feasible_source = deepcopy(infeasible)
    feasible_source["program"]["rhs"] = [q(0), q(0)]
    mutations.append(feasible_source)
    bare_unbounded = deepcopy(unbounded)
    bare_unbounded["recession_direction"] = None
    mutations.append(bare_unbounded)
    flat_direction = deepcopy(unbounded)
    flat_direction["recession_direction"] = [q(0), q(0)]
    mutations.append(flat_direction)
    bounded_source = deepcopy(unbounded)
    bounded_source["program"]["objective"] = [q(1), q(0)]
    mutations.append(bounded_source)

    for mutation in mutations:
        try:
            parsed = RationalLinearProgramResult.model_validate(mutation)
        except ValidationError:
            continue
        _assert_rejected_by_verifier(parsed)


@st.composite
def _diagonal_linear_programs(
    draw: st.DrawFn,
) -> tuple[dict[str, object], tuple[Fraction, ...], Fraction]:
    dimension = draw(st.integers(min_value=1, max_value=4))
    diagonal = draw(
        st.lists(
            st.integers(min_value=1, max_value=9),
            min_size=dimension,
            max_size=dimension,
        )
    )
    rhs = draw(
        st.lists(
            st.integers(min_value=0, max_value=9),
            min_size=dimension,
            max_size=dimension,
        )
    )
    objective = draw(
        st.lists(
            st.integers(min_value=-9, max_value=9),
            min_size=dimension,
            max_size=dimension,
        )
    )
    point = tuple(Fraction(rhs[index], diagonal[index]) for index in range(dimension))
    optimum = sum(
        (Fraction(objective[index]) * point[index] for index in range(dimension)),
        Fraction(),
    )
    coefficients = [
        [q(diagonal[row]) if row == column else q(0) for column in range(dimension)]
        for row in range(dimension)
    ]
    return (
        {
            "variables": [f"x{index}" for index in range(dimension)],
            "objective": [q(value) for value in objective],
            "coefficients": coefficients,
            "rhs": [q(value) for value in rhs],
        },
        point,
        optimum,
    )


@given(_diagonal_linear_programs())
@settings(max_examples=30, deadline=None)
def test_diagonal_linear_program_property(
    case: tuple[dict[str, object], tuple[Fraction, ...], Fraction],
) -> None:
    program, expected_point, expected_optimum = case
    result = _run_linear_program(program)

    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None
    assert (
        tuple(value.as_fraction() for value in result.primal_candidate)
        == expected_point
    )
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == expected_optimum


def test_linear_program_admission_is_result_sensitive_but_bounds_all_bases() -> None:
    maximum_shape = {
        "variables": [f"x{index}" for index in range(32)],
        "objective": [q(0)] * 32,
        "coefficients": [[q(0)] * 32 for _ in range(64)],
        "rhs": [q(0)] * 64,
    }
    assert StandardFormRationalLinearProgram.model_validate(maximum_shape)

    excessive_work = {
        "variables": [f"x{index}" for index in range(8)],
        "objective": [q(0)] * 8,
        "coefficients": [[q(1)] * 8 for _ in range(8)],
        "rhs": [q(0)] * 8,
    }
    with linear_validation_error():
        StandardFormRationalLinearProgram.model_validate(excessive_work)
