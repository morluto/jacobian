"""Typed polynomial expression normalization tests."""

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
    domain: str, expression: dict[str, Any]
) -> PolynomialExpressionNormalizeRequest:
    return PolynomialExpressionNormalizeRequest.model_validate(
        {"coefficient_domain": domain, "variables": ["x"], "expression": expression}
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
        normalize_polynomial_expression(request)
