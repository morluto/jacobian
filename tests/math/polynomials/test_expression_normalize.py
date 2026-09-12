"""Typed polynomial expression normalization tests."""

from fractions import Fraction
from math import comb, gcd, prod
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


def test_nested_cancellation_does_not_accumulate_zero_denominators() -> None:
    """Zero children must not contribute their cancelled denominators to a parent."""

    primes = (
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
    )
    base = prod(primes) * 10**93
    denominators = tuple(base * multiplier + 1 for multiplier in range(40, 124))
    assert all(len(str(denominator)) == 128 for denominator in denominators)
    assert all(
        gcd(left, right) == 1
        for index, left in enumerate(denominators)
        for right in denominators[index + 1 :]
    )

    def cancelling_pair(denominator: int) -> dict[str, Any]:
        literal = {"kind": "LITERAL", "value": {"num": 1, "den": denominator}}
        opposite = {
            "kind": "LITERAL",
            "value": {"num": -1, "den": denominator},
        }
        return {"kind": "ADD", "operands": [literal, opposite]}

    expression = {
        "kind": "ADD",
        "operands": [
            *(cancelling_pair(denominator) for denominator in denominators[:63]),
            {
                "kind": "ADD",
                "operands": [
                    cancelling_pair(denominator) for denominator in denominators[63:]
                ],
            },
        ],
    }

    result = normalize_polynomial_expression(_request("QQ", expression))
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


def test_non_literal_exact_cancellations_are_admitted() -> None:
    primes = (10**127 + 39, 10**127 + 79, 10**127 + 121)

    def cancelling_powers(prime: int) -> dict[str, Any]:
        return {
            "kind": "ADD",
            "operands": [
                {
                    "kind": "POWER",
                    "base": {"kind": "LITERAL", "value": {"num": 1, "den": prime}},
                    "exponent": 31,
                },
                {
                    "kind": "POWER",
                    "base": {"kind": "LITERAL", "value": {"num": -1, "den": prime}},
                    "exponent": 31,
                },
            ],
        }

    expression = {
        "kind": "ADD",
        "operands": [cancelling_powers(prime) for prime in primes],
    }
    result = normalize_polynomial_expression(_request("QQ", expression))
    assert result.polynomial.polynomial.terms == ()


def test_heterogeneous_coefficient_heights_use_aggregate_digits() -> None:
    product: dict[str, Any] = {"kind": "LITERAL", "value": {"num": 1, "den": 1}}
    variables = tuple(f"x{index}" for index in range(7))
    for name in variables:
        product = {
            "kind": "MULTIPLY",
            "operands": [
                product,
                {
                    "kind": "ADD",
                    "operands": [
                        {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                        {"kind": "VARIABLE", "name": name},
                        {
                            "kind": "POWER",
                            "base": {"kind": "VARIABLE", "name": name},
                            "exponent": 2,
                        },
                    ],
                },
            ],
        }
    expression = {
        "kind": "ADD",
        "operands": [
            {
                "kind": "POWER",
                "base": {"kind": "LITERAL", "value": {"num": 10**127, "den": 1}},
                "exponent": 32,
            },
            product,
        ],
    }
    result = normalize_polynomial_expression(
        _request("ZZ", expression, variables=variables)
    )
    assert len(result.polynomial.polynomial.terms) == 3**7
    constant = next(
        term.coefficient.as_fraction()
        for term in result.polynomial.polynomial.terms
        if term.exponents == (0,) * 7
    )
    assert constant == 10 ** (127 * 32) + 1


def _dense_quadratic_product(variables: tuple[str, ...]) -> dict[str, Any]:
    product: dict[str, Any] = {"kind": "LITERAL", "value": {"num": 1, "den": 1}}
    for name in variables:
        product = {
            "kind": "MULTIPLY",
            "operands": [
                product,
                {
                    "kind": "ADD",
                    "operands": [
                        {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                        {"kind": "VARIABLE", "name": name},
                        {
                            "kind": "POWER",
                            "base": {"kind": "VARIABLE", "name": name},
                            "exponent": 2,
                        },
                    ],
                },
            ],
        }
    return product


def test_overlapping_rational_addends_include_common_denominator_scaling() -> None:
    variables = tuple(f"x{index}" for index in range(6))
    product = _dense_quadratic_product(variables)
    primes = (10**127 + 39, 10**127 + 79)
    expression = {
        "kind": "ADD",
        "operands": [
            {
                "kind": "MULTIPLY",
                "operands": [
                    product,
                    {
                        "kind": "POWER",
                        "base": {
                            "kind": "LITERAL",
                            "value": {"num": 1, "den": prime},
                        },
                        "exponent": 20,
                    },
                ],
            }
            for prime in primes
        ],
    }
    with pytest.raises(
        OperationResourceAdmissionError,
        match="representation envelope",
    ):
        normalize_polynomial_expression(_request("QQ", expression, variables=variables))


def test_disjoint_monomial_denominators_are_not_globally_cleared() -> None:
    primes = (10**127 + 39, 10**127 + 79, 10**127 + 121)
    powers = (1, 2, 3)
    expression = {
        "kind": "ADD",
        "operands": [
            {
                "kind": "MULTIPLY",
                "operands": [
                    {
                        "kind": "POWER",
                        "base": {"kind": "VARIABLE", "name": "x"},
                        "exponent": power,
                    },
                    {
                        "kind": "POWER",
                        "base": {
                            "kind": "LITERAL",
                            "value": {"num": 1, "den": prime},
                        },
                        "exponent": 31,
                    },
                ],
            }
            for power, prime in zip(powers, primes, strict=True)
        ],
    }
    result = normalize_polynomial_expression(_request("QQ", expression))
    assert len(result.polynomial.polynomial.terms) == 3


def test_nested_constant_powers_are_capped_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractions import Fraction as ExactFraction

    original_pow = ExactFraction.__pow__

    def fail_huge_power(self: ExactFraction, exponent: object) -> ExactFraction:
        if isinstance(exponent, int) and exponent == 32 and abs(self.numerator) > 10**200:
            raise AssertionError("exact constant power evaluated past the digit envelope")
        return original_pow(self, exponent)

    monkeypatch.setattr(ExactFraction, "__pow__", fail_huge_power)
    expression = {
        "kind": "POWER",
        "base": {
            "kind": "POWER",
            "base": {"kind": "LITERAL", "value": {"num": 10**127, "den": 1}},
            "exponent": 32,
        },
        "exponent": 32,
    }
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(_request("ZZ", expression))


def test_powered_disjoint_sum_preserves_child_denominators() -> None:
    first = 10**127 + 39
    second = 10**127 + 79
    expression = {
        "kind": "POWER",
        "base": {
            "kind": "ADD",
            "operands": [
                {
                    "kind": "MULTIPLY",
                    "operands": [
                        {"kind": "VARIABLE", "name": "x"},
                        {
                            "kind": "POWER",
                            "base": {
                                "kind": "LITERAL",
                                "value": {"num": 1, "den": first},
                            },
                            "exponent": 32,
                        },
                    ],
                },
                {
                    "kind": "MULTIPLY",
                    "operands": [
                        {"kind": "VARIABLE", "name": "y"},
                        {
                            "kind": "POWER",
                            "base": {
                                "kind": "LITERAL",
                                "value": {"num": 1, "den": second},
                            },
                            "exponent": 32,
                        },
                    ],
                },
            ],
        },
        "exponent": 9,
    }
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(_request("QQ", expression, variables=("x", "y")))


def test_constant_add_caps_denominators_before_fraction_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_add = Fraction.__add__

    def fail_huge_add(self: Fraction, other: object) -> Fraction:
        if (
            isinstance(other, Fraction)
            and self.denominator.bit_length() > 8_000
            and other.denominator.bit_length() > 8_000
        ):
            raise AssertionError("unadmitted constant sum materialized")
        return original_add(self, other)

    monkeypatch.setattr(Fraction, "__add__", fail_huge_add)
    primes = [10**127 + 3 + 2 * index for index in range(8)]
    expression = {
        "kind": "ADD",
        "operands": [
            {
                "kind": "POWER",
                "base": {
                    "kind": "LITERAL",
                    "value": {"num": 1, "den": prime},
                },
                "exponent": 32,
            }
            for prime in primes
        ],
    }
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(_request("QQ", expression))

