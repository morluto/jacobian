"""Independent bounded coordinates retain exact source-coordinate LP evidence."""

from collections.abc import Sequence
from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.optimization import general_linear_program
from jacobian.math.optimization._general_models import (
    MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
    MAX_GENERAL_RATIONAL_INPUT_DIGITS,
    GeneralFormRationalLinearProgram,
    GeneralRationalLinearProgramResult,
    RationalObjectiveSense,
)


def _program(
    bounds: Sequence[tuple[Fraction | None, Fraction | None]],
    coefficients: list[Fraction],
    sense: RationalObjectiveSense = "MINIMIZE",
) -> GeneralFormRationalLinearProgram:
    return GeneralFormRationalLinearProgram.model_validate(
        {
            "variables": [
                {
                    "name": f"x{i}",
                    "lower_bound": CanonicalRational.from_fraction(lower)
                    if lower is not None
                    else None,
                    "upper_bound": CanonicalRational.from_fraction(upper)
                    if upper is not None
                    else None,
                }
                for i, (lower, upper) in enumerate(bounds)
            ],
            "objective": {
                "sense": sense,
                "coefficients": [
                    CanonicalRational.from_fraction(c) for c in coefficients
                ],
            },
            "constraints": [],
        }
    )


@pytest.mark.parametrize("size", [4, MAX_GENERAL_LINEAR_PROGRAM_VARIABLES])
def test_independent_box_is_admitted_through_the_source_variable_limit(
    size: int,
) -> None:
    program = _program([(Fraction(), Fraction(1))] * size, [Fraction(1)] * size)
    result = general_linear_program(program)
    assert result.status == "OPTIMAL"
    assert (
        result.primal_candidate == (CanonicalRational.from_fraction(Fraction()),) * size
    )
    assert result.primal_objective == CanonicalRational.from_fraction(Fraction())
    assert (
        result.lower_bound_dual
        == (CanonicalRational.from_fraction(Fraction(1)),) * size
    )
    assert result.program == program
    assert (
        GeneralRationalLinearProgramResult.model_validate_json(result.model_dump_json())
        == result
    )


@pytest.mark.parametrize("sense", ["MINIMIZE", "MAXIMIZE"])
def test_box_optimality_agrees_with_exhaustive_vertices(
    sense: RationalObjectiveSense,
) -> None:
    bounds = [
        (Fraction(-2), Fraction(3)),
        (Fraction(1, 3), Fraction(5, 2)),
        (Fraction(7), Fraction(7)),
    ]
    for raw in product((-1, 0, 1), repeat=3):
        coefficients = list(map(Fraction, raw))
        result = general_linear_program(_program(bounds, coefficients, sense))
        candidates = [
            sum(c * x for c, x in zip(coefficients, vertex, strict=True))
            for vertex in product(*bounds)
        ]
        optimum = min(candidates) if sense == "MINIMIZE" else max(candidates)
        assert result.status == "OPTIMAL"
        assert (
            result.primal_objective is not None
            and result.primal_objective.as_fraction() == optimum
        )
        assert result.dual_objective == result.primal_objective
        assert result.primal_candidate is not None
        assert (
            result.lower_bound_dual is not None and result.upper_bound_dual is not None
        )
        sign = 1 if sense == "MINIMIZE" else -1
        for c, point, lower_dual, upper_dual, (lower, upper) in zip(
            coefficients,
            result.primal_candidate,
            result.lower_bound_dual,
            result.upper_bound_dual,
            bounds,
            strict=True,
        ):
            x, ld, ud = (
                point.as_fraction(),
                lower_dual.as_fraction(),
                upper_dual.as_fraction(),
            )
            assert lower <= x <= upper
            assert ld + ud == c
            assert sign * ld >= 0 and sign * ud <= 0
            assert ld * (x - lower) == ud * (upper - x) == 0


@pytest.mark.parametrize("sense", ["MINIMIZE", "MAXIMIZE"])
def test_box_unbounded_direction_is_feasible_and_improves_objective(
    sense: RationalObjectiveSense,
) -> None:
    bounds = [
        (Fraction(0), Fraction(1)),
        (None, None),
        (None, Fraction(4)),
        (Fraction(2), None),
    ]
    coefficients = [Fraction(0), Fraction(-2), Fraction(3), Fraction(5)]
    result = general_linear_program(_program(bounds, coefficients, sense))
    assert result.status == "UNBOUNDED"
    assert (
        result.primal_candidate is not None and result.recession_direction is not None
    )
    point = [x.as_fraction() for x in result.primal_candidate]
    ray = [x.as_fraction() for x in result.recession_direction]
    slope = sum(c * d for c, d in zip(coefficients, ray, strict=True))
    assert slope < 0 if sense == "MINIMIZE" else slope > 0
    for x, d, (lower, upper) in zip(point, ray, bounds, strict=True):
        assert lower is None or (x >= lower and d >= 0)
        assert upper is None or (x <= upper and d <= 0)
    assert result.lower_bound_dual is None and result.upper_bound_dual is None
    assert result.dual_objective is None
    assert (
        GeneralRationalLinearProgramResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_zero_objective_with_free_and_one_sided_coordinates_is_optimal() -> None:
    bounds = [
        (None, None),
        (Fraction(2), None),
        (None, Fraction(-3)),
        (Fraction(7), Fraction(7)),
    ]
    result = general_linear_program(_program(bounds, [Fraction()] * len(bounds)))
    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None
    assert [x.as_fraction() for x in result.primal_candidate] == [0, 2, -3, 7]
    assert (
        result.primal_objective is not None
        and result.primal_objective.as_fraction() == 0
    )


def test_box_exact_objective_retains_growth_beyond_source_scalar_limit() -> None:
    size = MAX_GENERAL_LINEAR_PROGRAM_VARIABLES
    large = 10 ** (MAX_GENERAL_RATIONAL_INPUT_DIGITS - 1)
    endpoints = [Fraction(1, large + 2 * i + 1) for i in range(size)]
    coefficients = [Fraction(1, large + 2 * (size + i) + 1) for i in range(size)]
    program = _program([(x, x) for x in endpoints], coefficients)
    result = general_linear_program(program)
    expected = sum(c * x for c, x in zip(coefficients, endpoints, strict=True))
    assert expected.denominator >= 10**MAX_GENERAL_RATIONAL_INPUT_DIGITS
    assert result.status == "OPTIMAL"
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == expected
    assert result.dual_objective == result.primal_objective
    assert (
        GeneralRationalLinearProgramResult.model_validate_json(result.model_dump_json())
        == result
    )
