"""Tests for rational polynomial multiplication."""

import json
from fractions import Fraction

import pytest

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials._multiply_models import (
    RationalPolynomialMultiplyRequest,
    _maximum_product_coefficient_digits,
)
from jacobian.math.polynomials._multiply_ops import (
    compute_rational_polynomial_multiply as rational_polynomial_multiply,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _content_product_request(
    content_exponent: int,
) -> RationalPolynomialMultiplyRequest:
    denominator_base = 10**7000
    content = 10**content_exponent

    def operand(numerator: int, offsets: tuple[int, int]) -> RationalPolynomial:
        return RationalPolynomial(
            variables=("x",),
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        exponents=(1 - index,),
                        coefficient=CanonicalRational.from_fraction(
                            Fraction(numerator, denominator_base + offset)
                        ),
                    )
                    for index, offset in enumerate(offsets)
                )
            ),
        )

    return RationalPolynomialMultiplyRequest(
        left=operand(content, (1, 3)), right=operand(1, (7, 9))
    )


def test_rejects_collected_numerator_growth_before_multiplication() -> None:
    request = _content_product_request(20_000)
    # The middle coefficient has 34,001 numerator digits after reduction.
    a = 10**7000
    expected_middle = Fraction(10**20_000, (a + 1) * (a + 9)) + Fraction(
        10**20_000, (a + 3) * (a + 7)
    )
    assert (
        decimal_digit_width(expected_middle.numerator) > MAX_CANONICAL_RATIONAL_DIGITS
    )

    with pytest.raises(OperationDomainValidationError) as error:
        rational_polynomial_multiply(request)
    assert error.value.errors()[0]["type"] == "polynomial.invariant"


def test_admits_collected_numerator_growth_near_the_carrier_limit() -> None:
    content_exponent = MAX_CANONICAL_RATIONAL_DIGITS - 14_000 - 10
    request = _content_product_request(content_exponent)
    a = 10**7000
    content = 10**content_exponent
    expected = (
        Fraction(content, (a + 1) * (a + 7)),
        Fraction(content, (a + 1) * (a + 9)) + Fraction(content, (a + 3) * (a + 7)),
        Fraction(content, (a + 3) * (a + 9)),
    )

    result = rational_polynomial_multiply(request)

    assert tuple(
        term.coefficient.as_fraction() for term in result.polynomial.terms
    ) == (expected)
    assert (
        decimal_digit_width(expected[1].numerator) == MAX_CANONICAL_RATIONAL_DIGITS - 9
    )
    assert RationalPolynomial.model_validate_json(result.model_dump_json()) == result


def test_collected_component_bound_dominates_exact_fraction_convolution() -> None:
    """Check unequal denominator widths, signs, and common content independently."""

    def polynomial(coefficients: list[Fraction]) -> RationalPolynomial:
        return RationalPolynomial(
            variables=("x",),
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        exponents=(index,),
                        coefficient=CanonicalRational.from_fraction(coefficient),
                    )
                    for index, coefficient in reversed(list(enumerate(coefficients)))
                )
            ),
        )

    for content_exponent in (0, 5, 70):
        for denominator_exponent in (1, 8, 31):
            for count in range(1, 5):
                left = [
                    Fraction(
                        (-1) ** index * 10**content_exponent,
                        10**denominator_exponent + 2 * index + 1,
                    )
                    for index in range(count)
                ]
                right = [
                    Fraction(
                        index + 1, 10 ** (denominator_exponent + 1) + 2 * index + 3
                    )
                    for index in range(count + 1)
                ]
                bound = _maximum_product_coefficient_digits(
                    polynomial(left), polynomial(right)
                )
                expected = [Fraction(0) for _ in range(len(left) + len(right) - 1)]
                for i, a in enumerate(left):
                    for j, b in enumerate(right):
                        expected[i + j] += a * b
                assert all(
                    max(
                        decimal_digit_width(value.numerator),
                        decimal_digit_width(value.denominator),
                    )
                    <= bound
                    for value in expected
                )


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
    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": poly, "right": poly})
    )
    result = rational_polynomial_multiply(request)
    # Result should be x^2 + 2x + 1
    terms = result.polynomial.terms
    assert [
        (term.exponents, term.coefficient.num, term.coefficient.den) for term in terms
    ] == [((2,), 1, 1), ((1,), 2, 1), ((0,), 1, 1)]


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

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": left, "right": right})
    )
    with pytest.raises(OperationDomainValidationError, match="canonical term limit"):
        rational_polynomial_multiply(request)


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

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": polynomial, "right": polynomial})
    )
    result = rational_polynomial_multiply(request)

    assert len(result.polynomial.terms) == 129


def test_dense_backend_accepts_former_convolution_limit() -> None:
    terms = [
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [index]}
        for index in range(1024, -1, -1)
    ]
    polynomial = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {"terms": terms},
    }

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": polynomial, "right": polynomial})
    )
    result = rational_polynomial_multiply(request)
    assert len(result.polynomial.terms) == 2049
    assert result.polynomial.terms[1024].coefficient.num == 1025


def test_accumulated_coefficients_grow_by_the_collected_products_not_their_widths() -> (
    None
):
    """Collecting 64 products of 1/10^255 widens a coefficient by 510 digits.

    The old bound charged each collected product the *sum* of both operands'
    component widths, so this request was refused on a 32,770-digit estimate
    while every exact result here is a 511-digit rational, well inside the
    32,768-digit envelope. The exact product is what must be measured.
    """
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

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": polynomial, "right": polynomial})
    )
    result = rational_polynomial_multiply(request)

    # coefficient of x^63 collects exactly one product: 1/10^510
    assert result.polynomial.terms[0].coefficient.den == 10**510
    # coefficient of x^0 collects 64 of them: 64/10^510, reduced
    widest = max(
        max(
            decimal_digit_width(term.coefficient.num),
            decimal_digit_width(term.coefficient.den),
        )
        for term in result.polynomial.terms
    )
    assert widest == 511
    assert widest < MAX_CANONICAL_RATIONAL_DIGITS


def test_rejects_a_product_whose_exact_result_exceeds_the_digit_limit() -> None:
    """Two collected products of 1/10^16384 really do exceed the envelope.

    The square is 1/10^32768, one digit past the limit, so the refusal is a
    statement about the exact result rather than about operand widths.
    """
    coefficient = {"num": "1", "den": "1" + "0" * 16384}
    terms = [
        {"coefficient": coefficient, "exponents": [index]} for index in range(1, -1, -1)
    ]
    polynomial = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {"terms": terms},
    }

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": polynomial, "right": polynomial})
    )
    with pytest.raises(OperationDomainValidationError, match="coefficient digit limit"):
        rational_polynomial_multiply(request)


def test_accepts_large_product_within_term_and_digit_bounds() -> None:
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

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": left, "right": right})
    )
    result = rational_polynomial_multiply(request)

    assert len(result.polynomial.terms) == len(left_terms) * len(right_terms)


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

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": left, "right": right})
    )
    result = rational_polynomial_multiply(request)

    assert len(result.polynomial.terms) == 1025
    assert result.polynomial.terms[0].exponents == (2025,)
    assert result.polynomial.terms[-1].exponents == (1001,)
    assert result.polynomial.terms[0].coefficient.num == 10**256
    assert result.polynomial.terms[0].coefficient.den == 1


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

    request = RationalPolynomialMultiplyRequest.model_validate_json(json.dumps(payload))
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

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": monomial, "right": operand})
    )
    result = rational_polynomial_multiply(request)

    assert result.polynomial.terms[0].exponents == (1,)
    assert result.polynomial.terms[0].coefficient.num == 10 ** (
        MAX_CANONICAL_RATIONAL_DIGITS - 1
    )


def test_rejects_product_exponent_overflow() -> None:
    polynomial = {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [20_000]}]
        },
    }

    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": polynomial, "right": polynomial})
    )
    with pytest.raises(
        OperationDomainValidationError, match="canonical exponent limit"
    ):
        rational_polynomial_multiply(request)


@pytest.mark.parametrize("denominator", [1, 6])
def test_dense_sign_product_matches_integer_convolution(denominator: int) -> None:
    from fractions import Fraction

    n = 1001
    coefficients = [-1 if j % 3 == 0 else 1 for j in range(n)]
    polynomial = {
        "variables": ["z"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {
                        "num": str(coefficients[j]),
                        "den": str(denominator),
                    },
                    "exponents": [j],
                }
                for j in range(n - 1, -1, -1)
            ]
        },
    }
    request = RationalPolynomialMultiplyRequest.model_validate_json(
        json.dumps({"left": polynomial, "right": polynomial})
    )
    result = rational_polynomial_multiply(request)
    actual = {
        t.exponents[0]: t.coefficient.as_fraction() for t in result.polynomial.terms
    }
    expected = {
        k: Fraction(
            sum(
                coefficients[i] * coefficients[k - i]
                for i in range(max(0, k - n + 1), min(n - 1, k) + 1)
            ),
            denominator**2,
        )
        for k in range(2 * n - 1)
    }
    assert actual == {k: v for k, v in expected.items() if v}
