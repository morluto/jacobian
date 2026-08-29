"""Tests for polynomial GCD zero-operand edge cases (#2057)."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._models import PolynomialGcdRequest, PolynomialGcdResult
from jacobian.math.polynomials.operations import polynomial_gcd

ZERO = {
    "domain": "QQ",
    "variables": ["x"],
    "polynomial": {"terms": []},
}

F = {
    "domain": "QQ",
    "variables": ["x"],
    "polynomial": {
        "terms": [
            {"coefficient": {"num": "2", "den": "1"}, "exponents": [2]},
            {"coefficient": {"num": "-2", "den": "1"}, "exponents": [0]},
        ]
    },
}

G = {
    "domain": "QQ",
    "variables": ["x"],
    "polynomial": {
        "terms": [
            {"coefficient": {"num": "4", "den": "1"}, "exponents": [1]},
        ]
    },
}


def _run(request: PolynomialGcdRequest) -> PolynomialGcdResult:
    return polynomial_gcd(request.left, request.right)


def test_gcd_left_zero_succeeds() -> None:
    """gcd(0, f) = monic(f)."""
    result = _run(PolynomialGcdRequest.model_validate({"left": ZERO, "right": F}))
    assert len(result.gcd.polynomial.terms) == 2
    # monic of 2x^2 - 2 is x^2 - 1
    terms = sorted(result.gcd.polynomial.terms, key=lambda t: t.exponents[0])
    assert terms[0].coefficient.num == "-1"
    assert terms[0].coefficient.den == "1"
    assert terms[1].coefficient.num == "1"
    assert terms[1].coefficient.den == "1"


def test_gcd_right_zero_succeeds() -> None:
    """gcd(f, 0) = monic(f)."""
    result = _run(PolynomialGcdRequest.model_validate({"left": F, "right": ZERO}))
    assert len(result.gcd.polynomial.terms) == 2
    terms = sorted(result.gcd.polynomial.terms, key=lambda t: t.exponents[0])
    assert terms[0].coefficient.num == "-1"
    assert terms[1].coefficient.num == "1"


def test_gcd_is_symmetric() -> None:
    """gcd(f, 0) and gcd(0, f) return the same normalized GCD."""
    r1 = _run(PolynomialGcdRequest.model_validate({"left": ZERO, "right": F}))
    r2 = _run(PolynomialGcdRequest.model_validate({"left": F, "right": ZERO}))
    assert r1.gcd == r2.gcd


def test_both_zero_rejected() -> None:
    """gcd(0, 0) is rejected because zero has no monic normalization."""
    request = PolynomialGcdRequest.model_validate({"left": ZERO, "right": ZERO})
    with pytest.raises(OperationDomainValidationError, match="monic"):
        _run(request)


def test_bezout_left_zero() -> None:
    """Bézout identity: s*0 + t*f = gcd(f)."""
    result = _run(PolynomialGcdRequest.model_validate({"left": ZERO, "right": F}))
    # left_multiplier should be zero
    assert len(result.bezout.left_multiplier.polynomial.terms) == 0


def test_bezout_right_zero() -> None:
    """Bézout identity: s*f + t*0 = gcd(f)."""
    result = _run(PolynomialGcdRequest.model_validate({"left": F, "right": ZERO}))
    # right_multiplier should be zero
    assert len(result.bezout.right_multiplier.polynomial.terms) == 0


def test_gcd_with_nonzero_coprime() -> None:
    """Control: gcd of two nonzero coprime polynomials."""
    result = _run(PolynomialGcdRequest.model_validate({"left": F, "right": G}))
    # gcd(2x^2-2, 4x) = 2 (constant), monic = 1
    assert len(result.gcd.polynomial.terms) == 1
    assert result.gcd.polynomial.terms[0].coefficient.num == "1"
    assert result.gcd.polynomial.terms[0].exponents == (0,)


def test_gcd_negative_leading_coefficient() -> None:
    """GCD works with negative leading coefficient."""
    f = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "-3", "den": "1"}, "exponents": [2]},
                {"coefficient": {"num": "3", "den": "1"}, "exponents": [0]},
            ]
        },
    }
    result = _run(PolynomialGcdRequest.model_validate({"left": f, "right": ZERO}))
    # monic of -3x^2 + 3 = x^2 - 1
    terms = sorted(result.gcd.polynomial.terms, key=lambda t: t.exponents[0])
    assert terms[0].coefficient.num == "-1"
    assert terms[1].coefficient.num == "1"


def test_rational_leading_coefficient() -> None:
    """GCD works with rational leading coefficient."""
    f = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "2"}, "exponents": [1]},
                {"coefficient": {"num": "1", "den": "2"}, "exponents": [0]},
            ]
        },
    }
    result = _run(PolynomialGcdRequest.model_validate({"left": f, "right": ZERO}))
    # monic of (1/2)x + (1/2) = x + 1
    assert len(result.gcd.polynomial.terms) == 2
    terms = sorted(result.gcd.polynomial.terms, key=lambda t: t.exponents[0])
    assert terms[0].coefficient.num == "1"
    assert terms[1].coefficient.num == "1"
