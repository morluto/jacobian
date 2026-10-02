"""Use original-source admission for bounded Hermite-derived logarithmic data."""

import pytest
from sympy import Rational, Symbol, cancel, diff

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.rational_functions.operations import (
    formal_antiderivative,
    hermite_reduction,
    logarithmic_differential,
    partial_fractions,
)
from jacobian.math.polynomials.values import RationalFunction


@pytest.mark.parametrize("axis", ("x", "t"))
@pytest.mark.parametrize("sign", (-1, 1))
def test_original_source_admits_exact_derived_remainder(axis: str, sign: int) -> None:
    x = Symbol(axis)
    expression = sign * x / (99 * (x - 9) ** 2 * (x + 1))
    source = rational_function_from_sympy(expression, (axis,))
    expected_r = rational_function_from_sympy(-sign / (110 * (x - 9)), (axis,))
    expected_h = rational_function_from_sympy(sign / (990 * (x - 9) * (x + 1)), (axis,))
    rational_part, remainder = hermite_reduction(source)
    assert rational_part == expected_r
    assert remainder == expected_h
    assert (
        RationalFunction.model_validate_json(remainder.model_dump_json()) == expected_h
    )
    profile = partial_fractions(source)
    assert profile.hermite_remainder == expected_h
    logarithmic = logarithmic_differential(source)
    assert logarithmic.source == expected_h
    assert logarithmic.reconstructed == expected_h
    rows = {
        str(
            rational_polynomial_to_sympy(term.factor).as_expr()
        ): rational_polynomial_to_sympy(term.numerator).as_expr()
        for term in logarithmic.terms
    }
    assert rows == {
        f"{axis} - 9": Rational(sign, 9900),
        f"{axis} + 1": Rational(-sign, 9900),
    }
    formal = formal_antiderivative(source)
    assert formal.source == source
    assert formal.rational_part == expected_r
    assert formal.logarithmic_part == logarithmic
    assert (
        cancel(
            diff(rational_function_to_sympy(formal.rational_part), x)
            + rational_function_to_sympy(logarithmic.reconstructed)
            - expression
        )
        == 0
    )
    for value in (logarithmic, formal):
        assert type(value).model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize(
    "kind",
    ("zero", "polynomial", "double_pole", "simple_pole", "irreducible", "improper"),
)
def test_original_profile_preserves_reconstruction_and_nested_source(kind: str) -> None:
    x = Symbol("x")
    expression = {
        "zero": 0,
        "polynomial": x**3 + 2 * x,
        "double_pole": 1 / (x - 1) ** 2,
        "simple_pole": 1 / (x - 1),
        "irreducible": 1 / ((x * x + 1) * (x + 1)),
        "improper": (x**6 + 1) / ((x - 1) ** 2 * (x + 1)),
    }[kind]
    source = rational_function_from_sympy(expression, ("x",))
    rational_part, remainder = hermite_reduction(source)
    logarithmic = logarithmic_differential(source)
    formal = formal_antiderivative(source)
    assert logarithmic.source == (remainder if remainder.numerator.terms else source)
    assert logarithmic.reconstructed == remainder
    assert formal.logarithmic_part.source == remainder
    assert formal.logarithmic_part.reconstructed == remainder
    assert formal.rational_part == rational_part
    assert (
        cancel(
            diff(rational_function_to_sympy(rational_part), x)
            + rational_function_to_sympy(remainder)
            - expression
        )
        == 0
    )


def test_repeated_nonlinear_factor_stays_outside_linear_repeated_pole_envelope() -> (
    None
):
    x = Symbol("x")
    source = rational_function_from_sympy(1 / (x * x + 1) ** 2, ("x",))
    for operation in (
        partial_fractions,
        logarithmic_differential,
        formal_antiderivative,
    ):
        with pytest.raises(OperationResourceAdmissionError) as error:
            operation(source)
        assert error.value.errors()[0]["type"] == "polynomial.partial_fraction_budget"


def test_three_digit_original_source_still_requires_its_own_admission() -> None:
    x = Symbol("x")
    source = rational_function_from_sympy(1 / (100 * (x - 1)), ("x",))
    for operation in (logarithmic_differential, formal_antiderivative):
        with pytest.raises(OperationResourceAdmissionError) as error:
            operation(source)
        assert error.value.errors()[0]["type"] == "polynomial.partial_fraction_budget"
