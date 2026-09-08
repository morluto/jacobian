"""Chain coordinates preserve source-coordinate primal and dual relations."""

import json
from fractions import Fraction
from itertools import pairwise

import pytest

from jacobian.math.optimization import check_linear_optimality, general_linear_program
from jacobian.math.optimization._general_models import (
    GeneralFormRationalLinearProgram,
    GeneralRationalLinearProgramResult,
)
from jacobian.math.optimization._optimality import RationalLinearOptimalityCandidate


def _program(
    size: int = 9,
    *,
    reverse: bool = False,
    total: Fraction | None = Fraction(1),
    lower: Fraction = Fraction(1, 10),
) -> GeneralFormRationalLinearProgram:
    def q(value: Fraction | int) -> dict[str, str]:
        value = Fraction(value)
        return {"num": str(value.numerator), "den": str(value.denominator)}

    axis = list(reversed(range(size))) if reverse else list(range(size))
    rows = []
    for i in range(size - 1):
        coefficients = [int(j == i) - int(j == i + 1) for j in axis]
        if reverse:
            coefficients = [-c for c in coefficients]
        rows.append(
            {
                "label": f"order_{size - i}",
                "coefficients": list(map(q, coefficients)),
                "relation": "GE" if reverse else "LE",
                "rhs": q(0),
            }
        )
    if total is not None:
        rows.insert(
            size // 2,
            {
                "label": "total",
                "coefficients": [q(1)] * size,
                "relation": "EQ",
                "rhs": q(total),
            },
        )
    payload = {
        "variables": [{"name": f"x{j}", "lower_bound": q(lower)} for j in axis],
        "objective": {
            "sense": "MAXIMIZE",
            "coefficients": [q(j < size // 2) for j in axis],
        },
        "constraints": list(reversed(rows)) if reverse else rows,
    }
    return GeneralFormRationalLinearProgram.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("reverse", [False, True])
def test_ordered_simplex_recovers_original_optimum_and_dual(reverse: bool) -> None:
    source = _program(reverse=reverse)
    result = general_linear_program(source)
    assert result.status == "OPTIMAL"
    assert result.primal_objective is not None
    assert result.primal_candidate is not None
    assert result.stationarity_residuals is not None
    assert result.primal_objective.as_fraction() == Fraction(4, 9)
    assert all(
        value.as_fraction() == Fraction(1, 9) for value in result.primal_candidate
    )
    assert all(value.num == 0 for value in result.stationarity_residuals)
    decoded = GeneralRationalLinearProgramResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded.program == source
    assert decoded.primal_candidate is not None
    assert decoded.constraint_dual is not None
    assert decoded.lower_bound_dual is not None
    assert decoded.upper_bound_dual is not None
    candidate = RationalLinearOptimalityCandidate(
        program=decoded.program,
        primal_candidate=decoded.primal_candidate,
        constraint_dual=decoded.constraint_dual,
        lower_bound_dual=decoded.lower_bound_dual,
        upper_bound_dual=decoded.upper_bound_dual,
    )
    assert check_linear_optimality(candidate).is_optimal


@pytest.mark.parametrize("reverse", [False, True])
def test_chain_infeasibility_has_source_farkas_relation(reverse: bool) -> None:
    source = _program(reverse=reverse, total=Fraction(1, 2))
    result = general_linear_program(source)
    assert result.status == "INFEASIBLE"
    assert result.farkas_constraints is not None
    assert result.farkas_lower_bounds is not None
    multipliers = [v.as_fraction() for v in result.farkas_constraints]
    lower = [v.as_fraction() for v in result.farkas_lower_bounds]
    for column in range(9):
        assert (
            sum(
                row.coefficients[column].as_fraction() * y
                for row, y in zip(source.constraints, multipliers, strict=True)
            )
            + lower[column]
            == 0
        )
    assert all(v <= 0 for v in lower)
    assert all(
        y >= 0 if row.relation == "LE" else y <= 0
        for row, y in zip(source.constraints, multipliers, strict=True)
        if row.relation != "EQ"
    )
    assert (
        sum(
            row.rhs.as_fraction() * y
            for row, y in zip(source.constraints, multipliers, strict=True)
        )
        + sum(lower) / 10
        < 0
    )


def test_chain_unbounded_ray_preserves_order_and_improves_objective() -> None:
    result = general_linear_program(_program(total=None))
    assert result.status == "UNBOUNDED"
    assert result.recession_direction is not None
    direction = [v.as_fraction() for v in result.recession_direction]
    assert direction[0] >= 0
    assert all(a <= b for a, b in pairwise(direction))
    assert sum(direction[:4]) > 0


def test_unequal_lower_bounds_retain_general_normalization() -> None:
    source = _program(size=3, lower=Fraction(0))
    payload = source.model_dump(mode="json")
    payload["variables"][2]["lower_bound"] = {"num": "1", "den": "2"}
    source = GeneralFormRationalLinearProgram.model_validate_json(json.dumps(payload))
    result = general_linear_program(source)
    assert result.status == "OPTIMAL"
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == Fraction(1, 4)


def test_ordered_simplex_reaches_source_variable_boundary() -> None:
    source = _program(size=32, lower=Fraction(1, 33), reverse=True)
    result = general_linear_program(source)
    assert result.status == "OPTIMAL"
    assert result.primal_objective is not None
    assert result.primal_candidate is not None
    assert result.primal_objective.as_fraction() == Fraction(1, 2)
    assert all(
        value.as_fraction() == Fraction(1, 32) for value in result.primal_candidate
    )


@pytest.mark.parametrize("sense", ["MINIMIZE", "MAXIMIZE"])
def test_chain_objectives_match_independent_vertices(sense: str) -> None:
    from random import Random

    rng = Random(3516)
    for _ in range(20):
        coefficients = [rng.randrange(-7, 8) for _ in range(5)]
        payload = _program(size=5).model_dump(mode="json")
        payload["objective"] = {
            "sense": sense,
            "coefficients": [{"num": str(v), "den": "1"} for v in coefficients],
        }
        source = GeneralFormRationalLinearProgram.model_validate_json(
            json.dumps(payload)
        )
        # The vertices consist of a lower-bound prefix and a constant suffix.
        values = [
            sum(
                Fraction(1, 10) * coefficients[i]
                if i < split
                else (1 - Fraction(split, 10)) * coefficients[i] / (5 - split)
                for i in range(5)
            )
            for split in range(5)
        ]
        expected = (min if sense == "MINIMIZE" else max)(values)
        result = general_linear_program(source)
        assert result.status == "OPTIMAL"
        assert result.primal_objective is not None
        assert result.primal_objective.as_fraction() == expected
