"""Tests for rational polynomial multiplication."""

from jacobian.math.polynomials._multiply_operations import rational_polynomial_multiply
from jacobian.math.polynomials._multiply_models import RationalPolynomialMultiplyRequest
from jacobian.math.polynomials.values import RationalPolynomial


def test_multiply_x_plus_1() -> None:
    """(x+1) * (x+1) = x^2 + 2x + 1"""
    c1 = {"num": "1", "den": "1"}
    c2 = {"num": "2", "den": "1"}
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
    request = RationalPolynomialMultiplyRequest.model_validate({"left": poly, "right": poly})
    result = rational_polynomial_multiply(request)
    # Result should be x^2 + 2x + 1
    terms = result.polynomial.terms
    # Check we have 3 terms
    assert len(terms) == 3
