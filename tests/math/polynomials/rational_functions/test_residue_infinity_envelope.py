"""Infinity residues are Laurent coefficients, independent of finite factoring."""

from fractions import Fraction
from math import comb

import pytest
from sympy import Symbol

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.structured_operations import (
    residue_at_infinity,
)
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
    MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT,
    RationalFunction,
)


@pytest.mark.parametrize("axis", ("x", "t"))
@pytest.mark.parametrize(
    "kind",
    ("quartic_zero", "quartic_pole", "height", "full_degree", "polynomial", "zero"),
)
def test_proper_and_polynomial_residues_have_independent_known_answers(
    axis: str, kind: str
) -> None:
    x = Symbol(axis)
    bound = MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT
    expressions = {
        "quartic_zero": (1 / (x**4 + 1), Fraction(0)),
        "quartic_pole": (x**3 / (x**4 + 1), Fraction(-1)),
        "height": (1 / (100 * (x - 1)), Fraction(-1, 100)),
        "full_degree": (x ** (bound - 1) / (x**bound + 1), Fraction(-1)),
        "polynomial": (x**bound + x + 1, Fraction(0)),
        "zero": (0, Fraction(0)),
    }
    expression, expected = expressions[kind]
    source = rational_function_from_sympy(expression, (axis,))
    actual = residue_at_infinity(source)
    assert actual.as_fraction() == expected
    assert CanonicalRational.model_validate_json(actual.model_dump_json()) == actual


def test_improper_quadratic_denominator_matches_combinatorial_laurent_coefficient() -> (
    None
):
    x = Symbol("x")
    source = rational_function_from_sympy(x**8 / (x * x + x + 2), ("x",))
    # [t^7]1/(1+t+2t²) = sum (-1)^(7-j) C(7-j,j) 2^j.
    # x^8/(x²+x+2)=x^6/(1+1/x+2/x²), and the residue negates [x^-1].
    expected = -sum((-1) ** (7 - j) * comb(7 - j, j) * 2**j for j in range(4))
    assert residue_at_infinity(source).as_fraction() == expected


def test_monic_long_division_retains_large_exact_scalar_output() -> None:
    x = Symbol("x")
    height = 10**MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS - 1
    degree = MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT
    source = rational_function_from_sympy(x**degree / (x - Fraction(1, height)), ("x",))
    actual = residue_at_infinity(source)
    # x^n/(x-a) has Laurent coefficient a^n at x^-1 by the geometric identity.
    assert actual.as_fraction() == Fraction(-1, height**degree)
    assert CanonicalRational.model_validate_json(actual.model_dump_json()) == actual


def test_common_factor_cancellation_does_not_change_the_laurent_coefficient() -> None:
    source = RationalFunction.model_validate_json(
        '{"variables":["x"],"numerator":{"terms":['
        '{"coefficient":{"num":"2","den":"1"},"exponents":[1]},'
        '{"coefficient":{"num":"2","den":"1"},"exponents":[0]}]},'
        '"denominator":{"terms":['
        '{"coefficient":{"num":"1","den":"1"},"exponents":[2]},'
        '{"coefficient":{"num":"-1","den":"1"},"exponents":[0]}]}}'
    )
    x = Symbol("x")
    reduced = rational_function_from_sympy(2 / (x - 1), ("x",))
    assert residue_at_infinity(source) == residue_at_infinity(reduced)
    assert residue_at_infinity(source).as_fraction() == -2


def test_native_residue_rejects_malformed_source_before_shortcuts() -> None:
    source = rational_function_from_sympy(0, ("x",))
    forged = source.model_copy(update={"variables": ("!",)})
    with pytest.raises(OperationDomainValidationError) as error:
        residue_at_infinity(forged)
    assert error.value.errors()[0]["type"] == "rational_function.residue_source"
