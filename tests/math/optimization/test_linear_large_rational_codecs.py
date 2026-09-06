"""Admitted exact LP values must not inherit Python's decimal-string ceiling."""

import sys
from collections.abc import Iterator
from fractions import Fraction
from random import Random

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.optimization import general_linear_program, linear_program
from jacobian.math.optimization._general_models import (
    GeneralFormRationalLinearProgram,
    GeneralRationalLinearProgramResult,
)
from jacobian.math.optimization._models import (
    RationalLinearProgramResult,
    StandardFormRationalLinearProgram,
)


@pytest.fixture
def default_integer_string_limit() -> Iterator[int]:
    previous = sys.get_int_max_str_digits()
    default = sys.int_info.default_max_str_digits
    sys.set_int_max_str_digits(default)
    try:
        yield default
        assert sys.get_int_max_str_digits() == default
    finally:
        sys.set_int_max_str_digits(previous)


@pytest.mark.parametrize("general", [False, True])
def test_exact_certificate_exceeds_interpreter_digit_limit(
    default_integer_string_limit: int,
    general: bool,
) -> None:
    random = Random(3194)
    one = CanonicalRational.from_integer_ratio(1, 1)
    rows = tuple(
        tuple(
            one
            if i == j
            else CanonicalRational.from_integer_ratio(
                1, random.randrange(10**127, 10**128)
            )
            for j in range(7)
        )
        for i in range(7)
    )
    program = StandardFormRationalLinearProgram(
        variables=tuple(f"x{i}" for i in range(7)),
        objective=(one,) * 7,
        coefficients=rows,
        rhs=(one,) * 7,
    )
    result: RationalLinearProgramResult | GeneralRationalLinearProgramResult
    if general:
        general_program = GeneralFormRationalLinearProgram.model_validate(
            {
                "variables": [
                    {
                        "name": name,
                        "lower_bound": CanonicalRational.from_integer_ratio(0, 1),
                    }
                    for name in program.variables
                ],
                "objective": {"sense": "MINIMIZE", "coefficients": program.objective},
                "constraints": [
                    {
                        "label": f"row{i}",
                        "relation": "EQ",
                        "coefficients": row,
                        "rhs": one,
                    }
                    for i, row in enumerate(rows)
                ],
            }
        )
        result = general_linear_program(general_program)
        dual = result.constraint_dual
    else:
        result = linear_program(program)
        dual = result.dual_candidate
    assert result.status == "OPTIMAL"
    assert result.primal_candidate is not None and dual is not None
    assert (
        max(len(format_canonical_integer(abs(v.num))) for v in result.primal_candidate)
        > default_integer_string_limit
    )
    x = tuple(v.as_fraction() for v in result.primal_candidate)
    y = tuple(v.as_fraction() for v in dual)
    assert all(v > 0 for v in (*x, *y))
    assert all(
        sum(a.as_fraction() * b for a, b in zip(row, x, strict=True)) == 1
        for row in rows
    )
    assert all(
        sum(rows[i][j].as_fraction() * y[i] for i in range(7)) == 1 for j in range(7)
    )
    assert sum(x) == sum(y)
    assert (
        result.primal_objective is not None
        and result.primal_objective.as_fraction() == sum(x)
    )
    assert result.model_validate_json(result.model_dump_json()) == result


def test_general_lp_normalized_rhs_and_source_point_exceed_interpreter_limit(
    default_integer_string_limit: int,
) -> None:
    random = Random(3194)
    lower = tuple(Fraction(1, random.randrange(10**127, 10**128)) for _ in range(20))
    coefficients = tuple(
        Fraction(1, random.randrange(10**127, 10**128)) for _ in range(20)
    )
    one = CanonicalRational.from_integer_ratio(1, 1)
    wire_coefficients = tuple(CanonicalRational.from_fraction(c) for c in coefficients)
    program = GeneralFormRationalLinearProgram.model_validate(
        {
            "variables": [
                {"name": f"x{i}", "lower_bound": CanonicalRational.from_fraction(v)}
                for i, v in enumerate(lower)
            ],
            "objective": {"sense": "MINIMIZE", "coefficients": wire_coefficients},
            "constraints": [
                {
                    "label": "weighted_sum",
                    "relation": "EQ",
                    "coefficients": wire_coefficients,
                    "rhs": one,
                }
            ],
        }
    )
    # The affine shift produces 1 - sum(c_i*l_i), beyond Python's default
    # decimal conversion limit even though every source component has <=128 digits.
    shifted = CanonicalRational.from_fraction(
        Fraction(1) - sum(c * v for c, v in zip(coefficients, lower, strict=True))
    )
    assert len(format_canonical_integer(shifted.den)) > default_integer_string_limit
    result = general_linear_program(program)
    assert result.status == "OPTIMAL" and result.primal_candidate is not None
    assert (
        max(len(format_canonical_integer(abs(v.num))) for v in result.primal_candidate)
        > default_integer_string_limit
    )
    x = tuple(v.as_fraction() for v in result.primal_candidate)
    assert all(v >= bound for v, bound in zip(x, lower, strict=True))
    assert sum(c * v for c, v in zip(coefficients, x, strict=True)) == 1
    assert result.primal_objective == result.dual_objective == one
    assert result.constraint_dual == (one,)
    assert result.model_validate_json(result.model_dump_json()) == result
