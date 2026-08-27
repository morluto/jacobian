"""Tests for rational polynomial multiplication."""

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.math.polynomials._multiply_models import RationalPolynomialMultiplyRequest
from jacobian.math.polynomials._multiply_operations import rational_polynomial_multiply


def test_multiply_x_plus_1() -> None:
    """(x+1) * (x+1) = x^2 + 2x + 1"""
    c1 = {"num": "1", "den": "1"}
    poly = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": c1, "exponents": [1]},
                {"coefficient": c1, "exponents": [0]},
            ],
        },
    }
    request = RationalPolynomialMultiplyRequest.model_validate(
        {"left": poly, "right": poly}
    )
    result = rational_polynomial_multiply(request)
    # Result should be x^2 + 2x + 1
    terms = result.polynomial.terms
    # Check we have 3 terms
    assert len(terms) == 3


def test_rejects_product_support_budget() -> None:
    left_terms = [
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [index, 0]}
        for index in range(64, -1, -1)
    ]
    right_terms = [
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, index]}
        for index in range(64, -1, -1)
    ]
    left = {
        "domain": "QQ",
        "variables": ["x", "y"],
        "polynomial": {"terms": left_terms},
    }
    right = {
        "domain": "QQ",
        "variables": ["x", "y"],
        "polynomial": {"terms": right_terms},
    }

    with pytest.raises(ValidationError, match="canonical term limit"):
        RationalPolynomialMultiplyRequest.model_validate({"left": left, "right": right})


def test_accepts_dense_univariate_product_with_compact_support() -> None:
    terms = [
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [index]}
        for index in range(64, -1, -1)
    ]
    polynomial = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {"terms": terms},
    }

    request = RationalPolynomialMultiplyRequest.model_validate(
        {"left": polynomial, "right": polynomial}
    )
    result = rational_polynomial_multiply(request)

    assert len(result.polynomial.terms) == 129


def test_rejects_excessive_convolution_work() -> None:
    terms = [
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [index]}
        for index in range(1024, -1, -1)
    ]
    polynomial = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {"terms": terms},
    }

    with pytest.raises(ValidationError, match="convolution work limit"):
        RationalPolynomialMultiplyRequest.model_validate(
            {"left": polynomial, "right": polynomial}
        )


def test_rejects_accumulated_coefficient_growth() -> None:
    coefficient = {"num": "1", "den": "1" + "0" * 255}
    terms = [
        {"coefficient": coefficient, "exponents": [index]}
        for index in range(63, -1, -1)
    ]
    polynomial = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {"terms": terms},
    }

    with pytest.raises(ValidationError, match="coefficient digit limit"):
        RationalPolynomialMultiplyRequest.model_validate(
            {"left": polynomial, "right": polynomial}
        )


def test_rejects_serialized_result_budget() -> None:
    coefficient = {"num": "9" * 256, "den": "1"}
    left_terms = [
        {"coefficient": coefficient, "exponents": [0, 0, index]}
        for index in range(3, -1, -1)
    ]
    right_terms = [
        {"coefficient": coefficient, "exponents": [x, y, 0]}
        for x in range(31, -1, -1)
        for y in range(31, -1, -1)
    ]
    left = {
        "domain": "QQ",
        "variables": ["x", "y", "z"],
        "polynomial": {"terms": left_terms},
    }
    right = {
        "domain": "QQ",
        "variables": ["x", "y", "z"],
        "polynomial": {"terms": right_terms},
    }

    with pytest.raises(ValidationError, match="serialized result size"):
        RationalPolynomialMultiplyRequest.model_validate({"left": left, "right": right})


def test_accepts_product_sensitive_operand_budgets() -> None:
    coefficient = {"num": "1" + "0" * 256, "den": "1"}
    left = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": coefficient, "exponents": [exponent]}
                for exponent in range(2025, 1000, -1)
            ]
        },
    }
    right = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
        },
    }

    request = RationalPolynomialMultiplyRequest.model_validate(
        {"left": left, "right": right}
    )
    result = rational_polynomial_multiply(request)

    assert len(result.polynomial.terms) == 1025
    assert result.polynomial.terms[0].exponents == (2025,)
    assert result.polynomial.terms[-1].exponents == (1001,)
    assert result.polynomial.terms[0].coefficient.num == coefficient["num"]
    assert result.polynomial.terms[0].coefficient.den == coefficient["den"]


@pytest.mark.parametrize("identity_on_left", [True, False])
def test_accepts_identity_product_at_coefficient_boundary(
    identity_on_left: bool,
) -> None:
    coefficient = {
        "num": "1" + "0" * (MAX_CANONICAL_RATIONAL_DIGITS - 1),
        "den": "1",
    }
    identity = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
        },
    }
    operand = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {"terms": [{"coefficient": coefficient, "exponents": [0]}]},
    }
    payload = (
        {"left": identity, "right": operand}
        if identity_on_left
        else {"left": operand, "right": identity}
    )

    request = RationalPolynomialMultiplyRequest.model_validate(payload)
    result = rational_polynomial_multiply(request)

    assert result == request.right if identity_on_left else result == request.left


def test_accepts_coefficient_one_monomial_shift_at_coefficient_boundary() -> None:
    coefficient = {
        "num": "1" + "0" * (MAX_CANONICAL_RATIONAL_DIGITS - 1),
        "den": "1",
    }
    monomial = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": [1],
                }
            ]
        },
    }
    operand = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {"terms": [{"coefficient": coefficient, "exponents": [0]}]},
    }

    request = RationalPolynomialMultiplyRequest.model_validate(
        {"left": monomial, "right": operand}
    )
    result = rational_polynomial_multiply(request)

    assert result.polynomial.terms[0].exponents == (1,)
    assert result.polynomial.terms[0].coefficient.num == coefficient["num"]


def test_rejects_product_exponent_overflow() -> None:
    polynomial = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [20_000]}]
        },
    }

    with pytest.raises(ValidationError, match="canonical exponent limit"):
        RationalPolynomialMultiplyRequest.model_validate(
            {"left": polynomial, "right": polynomial}
        )
