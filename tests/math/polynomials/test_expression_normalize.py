"""Typed polynomial expression normalization tests."""

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._expression_normalize import (
    PolynomialExpressionNormalizeRequest,
    PolynomialExpressionNormalizeResult,
    PolynomialExpressionSource,
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


def _normalize(
    request: PolynomialExpressionNormalizeRequest,
) -> PolynomialExpressionNormalizeResult:
    return normalize_polynomial_expression(
        PolynomialExpressionSource(
            coefficient_domain=request.coefficient_domain,
            variables=request.variables,
            expression=request.expression,
        )
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
    result = _normalize(request)
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
    assert _normalize(_request("QQ", literal)).polynomial
    with pytest.raises(OperationDomainValidationError):
        _normalize(_request("ZZ", literal))


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
    result = _normalize(request)
    assert result.polynomial.variables == ()
    assert result.polynomial.polynomial.terms == ()


def test_expression_result_round_trips_and_is_canonical() -> None:
    request = _request(
        "QQ",
        {
            "kind": "ADD",
            "operands": [
                {"kind": "VARIABLE", "name": "x"},
                {"kind": "LITERAL", "value": {"num": 1, "den": 2}},
                {"kind": "LITERAL", "value": {"num": 1, "den": 2}},
            ],
        },
    )
    result = _normalize(request)
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.polynomial.polynomial.terms[0].exponents == (1,)
    assert decoded.polynomial.polynomial.terms[1].coefficient.as_fraction() == 1


def test_grammar_rejects_division_negative_power_and_deep_raw_trees() -> None:
    with pytest.raises(ValidationError):
        _request(
            "QQ",
            {
                "kind": "DIVIDE",
                "numerator": {"kind": "VARIABLE", "name": "x"},
                "denominator": {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
            },
        )
    with pytest.raises(ValidationError):
        _request(
            "QQ",
            {
                "kind": "POWER",
                "base": {"kind": "VARIABLE", "name": "x"},
                "exponent": -1,
            },
        )

    expression: dict[str, Any] = {
        "kind": "VARIABLE",
        "name": "x",
    }
    for _ in range(64):
        expression = {"kind": "POWER", "base": expression, "exponent": 1}
    with pytest.raises(ValidationError, match="depth"):
        _request("QQ", expression)


def test_oversized_literal_is_a_typed_resource_rejection() -> None:
    with pytest.raises(OperationResourceAdmissionError) as error:
        _normalize(
            _request(
                "ZZ",
                {"kind": "LITERAL", "value": {"num": 10**129, "den": 1}},
            )
        )
    assert error.value.errors()[0]["type"] == "polynomial.expression.literal_bound"


def test_many_rational_denominators_are_admitted_conservatively() -> None:
    """Height admission accounts for denominator accumulation in additions."""

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
        _normalize(request)


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

    result = _normalize(_request("ZZ", tree(128)))
    assert (
        result.polynomial.polynomial.terms[0].coefficient.as_fraction() == 128 * 10**127
    )


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
        _normalize(request)
    assert error.value.errors()[0]["type"] == (
        "polynomial.expression.result_representation_bound"
    )
