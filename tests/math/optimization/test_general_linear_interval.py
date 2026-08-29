"""Exact interval presolve for one-variable general rational LPs."""

from fractions import Fraction

from jacobian.math.optimization._general_linear_program import (
    _INTERVAL_RESULT_DIGITS,
    _wire,
    general_linear_program,
)
from jacobian.math.optimization._general_models import (
    MAX_GENERAL_RATIONAL_INPUT_DIGITS,
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


def test_presolve_admits_513_digit_residuals_from_128_digit_inputs() -> None:
    """Endpoint division plus residual mul/sub can need 4*128+1 digits.

    With a 512-digit conversion ceiling the tall residual fails to wire, the
    one-variable presolve returns None, and a 32-inequality source then exceeds
    the slack-expanded variable bound instead of returning the exact optimum.
    """
    digit_bound = MAX_GENERAL_RATIONAL_INPUT_DIGITS
    offsets = (11509, 61478, 41469, 52773, 5591, 35311, 73132, 999)
    values = tuple(10**digit_bound - offset for offset in offsets)
    assert all(len(str(value)) == digit_bound for value in values)
    assert 4 * digit_bound + 1 == _INTERVAL_RESULT_DIGITS

    point = Fraction(values[2], values[3]) / Fraction(values[0], values[1])
    tall_residual = Fraction(values[4], values[5]) * point - Fraction(
        -values[6], values[7]
    )
    assert len(str(abs(tall_residual.numerator))) == 513
    assert _wire(tall_residual, max_digits=512) is None
    assert _wire(tall_residual, max_digits=_INTERVAL_RESULT_DIGITS) is not None

    constraints = [
        {
            "label": "active",
            "coefficients": [_q(values[0], values[1])],
            "relation": "GE",
            "rhs": _q(values[2], values[3]),
        },
        {
            "label": "tall",
            "coefficients": [_q(values[4], values[5])],
            "relation": "GE",
            "rhs": _q(-values[6], values[7]),
        },
        *[
            {
                "label": f"weak_{index}",
                "coefficients": [_q(1)],
                "relation": "GE",
                "rhs": _q(-1),
            }
            for index in range(30)
        ],
    ]
    program = GeneralFormRationalLinearProgram.model_validate(
        {
            "variables": [{"name": "a", "lower_bound": None, "upper_bound": None}],
            "objective": {"sense": "MINIMIZE", "coefficients": [_q(1)]},
            "constraints": constraints,
        }
    )

    result = general_linear_program(program)

    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None
    assert result.primal_candidate[0].as_fraction() == point
    assert result.constraint_slacks is not None
    assert result.constraint_slacks[1].as_fraction() == tall_residual
    assert (
        GeneralRationalLinearProgramResult.model_validate(
            result.model_dump(mode="json")
        )
        == result
    )
