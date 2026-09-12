"""Typed polynomial expression normalization tests."""

from fractions import Fraction
from math import comb
from typing import Any

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._expression_normalize import (
    PolynomialExpressionNormalizeRequest,
    normalize_polynomial_expression,
)


def _request(
    domain: str, expression: dict[str, Any], variables: tuple[str, ...] = ("x",)
) -> PolynomialExpressionNormalizeRequest:
    return PolynomialExpressionNormalizeRequest.model_validate(
        {
            "coefficient_domain": domain,
            "variables": list(variables),
            "expression": expression,
        }
    )


def test_binomial_square_normalizes_without_parsing_strings() -> None:
    request = _request(
        "ZZ",
        {
            "kind": "POWER",
            "base": {
                "kind": "ADD",
                "operands": [
                    {"kind": "VARIABLE", "name": "x"},
                    {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                ],
            },
            "exponent": 2,
        },
    )
    result = normalize_polynomial_expression(request)
    assert [term.exponents for term in result.polynomial.polynomial.terms] == [
        (2,),
        (1,),
        (0,),
    ]
    assert [
        term.coefficient.as_fraction() for term in result.polynomial.polynomial.terms
    ] == [1, 2, 1]


def test_qq_accepts_and_zz_rejects_nonintegral_literals() -> None:
    literal = {"kind": "LITERAL", "value": {"num": 1, "den": 2}}
    assert normalize_polynomial_expression(_request("QQ", literal)).polynomial
    with pytest.raises(OperationDomainValidationError):
        normalize_polynomial_expression(_request("ZZ", literal))


def test_constant_axis_and_exact_cancellation_are_preserved() -> None:
    request = PolynomialExpressionNormalizeRequest.model_validate(
        {
            "coefficient_domain": "ZZ",
            "variables": [],
            "expression": {
                "kind": "ADD",
                "operands": [
                    {"kind": "LITERAL", "value": {"num": 2, "den": 1}},
                    {"kind": "LITERAL", "value": {"num": -2, "den": 1}},
                ],
            },
        }
    )
    result = normalize_polynomial_expression(request)
    assert result.polynomial.variables == ()
    assert result.polynomial.polynomial.terms == ()


def test_many_distinct_rational_denominators_are_rejected() -> None:
    """Distinct denominators still charge their common-denominator growth."""

    def tree(start: int, count: int) -> dict[str, Any]:
        if count == 1:
            return {
                "kind": "LITERAL",
                "value": {"num": 1, "den": 10**127 + 2 * start + 1},
            }
        half = count // 2
        return {
            "kind": "ADD",
            "operands": [tree(start, half), tree(start + half, half)],
        }

    request = _request("QQ", tree(0, 128))
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(request)


def test_many_shared_nonunit_denominators_are_admitted() -> None:
    """A shared denominator is paid for once across a wide addition."""

    denominator = 10**127 + 1
    literal = {
        "kind": "LITERAL",
        "value": {"num": 1, "den": denominator},
    }

    def tree(count: int) -> dict[str, Any]:
        if count == 1:
            return literal
        half = count // 2
        return {
            "kind": "ADD",
            "operands": [tree(half), tree(half)],
        }

    result = normalize_polynomial_expression(_request("QQ", tree(128)))
    assert [
        term.coefficient.as_fraction() for term in result.polynomial.polynomial.terms
    ] == [Fraction(128, denominator)]


def test_many_integral_addends_use_per_coefficient_height() -> None:
    """Integral additions grow by carries, not by multiplying heights."""

    def tree(count: int) -> dict[str, Any]:
        if count == 1:
            return {
                "kind": "LITERAL",
                "value": {"num": 10**127, "den": 1},
            }
        half = count // 2
        return {
            "kind": "ADD",
            "operands": [tree(half), tree(half)],
        }

    result = normalize_polynomial_expression(_request("ZZ", tree(128)))
    assert (
        result.polynomial.polynomial.terms[0].coefficient.as_fraction() == 128 * 10**127
    )


def test_low_dimensional_power_uses_attainable_support() -> None:
    """A binomial power is represented by its 12 attainable monomials."""

    expression = {
        "kind": "MULTIPLY",
        "operands": [
            {
                "kind": "POWER",
                "base": {
                    "kind": "LITERAL",
                    "value": {"num": 10**127, "den": 1},
                },
                "exponent": 32,
            },
            {
                "kind": "POWER",
                "base": {
                    "kind": "ADD",
                    "operands": [
                        {"kind": "VARIABLE", "name": "x"},
                        {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                    ],
                },
                "exponent": 11,
            },
        ],
    }

    result = normalize_polynomial_expression(_request("ZZ", expression))
    scale = 10 ** (127 * 32)
    assert [term.exponents for term in result.polynomial.polynomial.terms] == [
        (exponent,) for exponent in range(11, -1, -1)
    ]
    assert [
        term.coefficient.as_fraction() for term in result.polynomial.polynomial.terms
    ] == [scale * comb(11, exponent) for exponent in range(11, -1, -1)]


def test_expansion_work_is_charged_separately_from_support() -> None:
    """A small one-variable result can still exceed convolution work."""

    expression = {
        "kind": "POWER",
        "base": {
            "kind": "ADD",
            "operands": [
                {"kind": "VARIABLE", "name": "x"},
                {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
            ],
        },
        "exponent": 32,
    }
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(_request("ZZ", expression))


def test_large_exact_result_is_rejected_before_expansion() -> None:
    """The normalized value must fit its complete exact representation envelope."""

    component = 10**127 + 123_456_789
    coefficient = {"num": component + 1, "den": component}

    def variable_power(variable: str, exponent: int) -> dict[str, Any]:
        return {
            "kind": "POWER",
            "base": {"kind": "VARIABLE", "name": variable},
            "exponent": exponent,
        }

    def factor(variable: str, exponent: int) -> dict[str, Any]:
        monomial = {
            "kind": "MULTIPLY",
            "operands": [
                {"kind": "LITERAL", "value": coefficient},
                variable_power(variable, exponent),
            ],
        }
        return {
            "kind": "ADD",
            "operands": [
                monomial,
                {"kind": "LITERAL", "value": coefficient},
            ],
        }

    expression = {
        "kind": "MULTIPLY",
        "operands": [
            factor(variable, exponent)
            for variable in ("x", "y", "z")
            for exponent in (1, 2, 4, 8)
        ],
    }
    request = _request("QQ", expression, variables=("x", "y", "z"))

    with pytest.raises(OperationResourceAdmissionError) as error:
        normalize_polynomial_expression(request)
    assert error.value.errors()[0]["type"] == (
        "polynomial.expression.result_representation_bound"
    )
