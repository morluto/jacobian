"""Tests for rational polynomial multiplication."""

import pytest
from pydantic import ValidationError

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


def test_rejects_product_term_budget() -> None:
    terms = [
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [index, 0]}
        for index in range(64, -1, -1)
    ]
    polynomial = {
        "domain": "QQ",
        "variables": ["x", "y"],
        "polynomial": {"terms": terms},
    }

    with pytest.raises(ValidationError, match="canonical term limit"):
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
