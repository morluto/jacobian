"""Exact interval presolve for one-variable general rational LPs."""

from fractions import Fraction

from jacobian.math.optimization._general_linear_program import general_linear_program
from jacobian.math.optimization._general_models import (
    GeneralFormRationalLinearProgram,
    GeneralRationalLinearProgramResult,
)


def _q(numerator: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(numerator), "den": str(denominator)}


def _lower_bound_program(
    thresholds: tuple[tuple[int, int], ...],
) -> GeneralFormRationalLinearProgram:
    return GeneralFormRationalLinearProgram.model_validate(
        {
            "variables": [{"name": "a", "lower_bound": _q(0), "upper_bound": None}],
            "objective": {"sense": "MINIMIZE", "coefficients": [_q(1)]},
            "constraints": [
                {
                    "label": f"bound_{index}",
                    "coefficients": [_q(1)],
                    "relation": "GE",
                    "rhs": _q(numerator, denominator),
                }
                for index, (numerator, denominator) in enumerate(thresholds)
            ],
        }
    )


def test_source_six_row_interval_program_is_solved_before_slack_expansion() -> None:
    program = _lower_bound_program(((2, 3), (3, 4), (7, 9), (3, 4), (5, 7), (2, 3)))

    result = general_linear_program(program)

    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None
    assert result.primal_candidate[0].as_fraction() == Fraction(7, 9)
    assert result.primal_objective is not None
    assert result.primal_objective.as_fraction() == Fraction(7, 9)
    assert result.constraint_dual is not None
    assert tuple(value.as_fraction() for value in result.constraint_dual) == (
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert (
        GeneralRationalLinearProgramResult.model_validate(
            result.model_dump(mode="json")
        )
        == result
    )


def test_distinct_four_row_interval_program_has_same_exact_optimum() -> None:
    result = general_linear_program(
        _lower_bound_program(((2, 3), (3, 4), (7, 9), (5, 7)))
    )

    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None
    assert result.primal_candidate[0].as_fraction() == Fraction(7, 9)
