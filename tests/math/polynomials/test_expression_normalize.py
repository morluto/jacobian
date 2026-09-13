"""Typed polynomial expression normalization tests."""

from fractions import Fraction
from math import comb, gcd, prod
from typing import Any

import pytest
import sympy

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._expression_normalize import (
    PolynomialExpressionNormalizeRequest,
    PolynomialMultiply,
    _ceil_log2,
    _metrics,
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


def test_repeated_binomial_product_uses_attainable_support() -> None:
    """Twelve copies of A*(1+x) have 4,096 paths but only 13 monomials."""

    expression = {
        "kind": "MULTIPLY",
        "operands": [
            {
                "kind": "ADD",
                "operands": [
                    {"kind": "LITERAL", "value": {"num": 10**127, "den": 1}},
                    {
                        "kind": "MULTIPLY",
                        "operands": [
                            {"kind": "LITERAL", "value": {"num": 10**127, "den": 1}},
                            {"kind": "VARIABLE", "name": "x"},
                        ],
                    },
                ],
            }
            for _ in range(12)
        ],
    }

    result = normalize_polynomial_expression(_request("ZZ", expression))
    assert [term.exponents for term in result.polynomial.polynomial.terms] == [
        (exponent,) for exponent in range(12, -1, -1)
    ]
    reference = sympy.expand((10**127) ** 12 * (1 + sympy.Symbol("x")) ** 12)
    assembled = sympy.expand(
        sum(
            term.coefficient.as_fraction().numerator
            * sympy.Symbol("x") ** term.exponents[0]
            for term in result.polynomial.polynomial.terms
        )
    )
    assert sympy.simplify(assembled - reference) == 0


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
        if (
            isinstance(exponent, int)
            and exponent == 32
            and abs(self.numerator) > 10**200
        ):
            raise AssertionError(
                "exact constant power evaluated past the digit envelope"
            )
        raised = original_pow(self, exponent)  # type: ignore[call-overload]
        assert isinstance(raised, ExactFraction)
        return raised

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
        normalize_polynomial_expression(
            _request("QQ", expression, variables=("x", "y"))
        )


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
        summed = original_add(self, other)  # type: ignore[call-overload]
        assert isinstance(summed, Fraction)
        return summed

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


def test_powered_univariate_disjoint_sum_accounts_for_colliding_dens() -> None:
    primes = tuple(10**127 + 39 + 210 * index for index in range(33))
    addends = [
        {
            "kind": "MULTIPLY",
            "operands": [
                {
                    "kind": "POWER",
                    "base": {"kind": "VARIABLE", "name": "x"},
                    "exponent": index,
                },
                {
                    "kind": "POWER",
                    "base": {
                        "kind": "LITERAL",
                        "value": {"num": 1, "den": prime},
                    },
                    "exponent": 32,
                },
            ],
        }
        for index, prime in enumerate(primes)
    ]
    expression = {
        "kind": "POWER",
        "base": {"kind": "ADD", "operands": addends},
        "exponent": 2,
    }
    with pytest.raises(OperationResourceAdmissionError, match="coefficient-height"):
        normalize_polynomial_expression(_request("QQ", expression))


def test_constant_product_skips_exact_fractions_past_the_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_mul = Fraction.__mul__

    def fail_huge_mul(self: Fraction, other: object) -> Fraction:
        if isinstance(other, Fraction) and (
            self.denominator.bit_length() > 20_000
            or other.denominator.bit_length() > 20_000
        ):
            raise AssertionError("unadmitted constant product materialized")
        product = original_mul(self, other)  # type: ignore[call-overload]
        assert isinstance(product, Fraction)
        return product

    monkeypatch.setattr(Fraction, "__mul__", fail_huge_mul)
    primes = [10**127 + 39 + 210 * index for index in range(108)]
    groups = [
        {
            "kind": "MULTIPLY",
            "operands": [
                {
                    "kind": "POWER",
                    "base": {
                        "kind": "LITERAL",
                        "value": {"num": 1, "den": primes[start + offset]},
                    },
                    "exponent": 32,
                }
                for offset in range(3)
            ],
        }
        for start in range(0, 108, 3)
    ]
    expression = {"kind": "MULTIPLY", "operands": groups}
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(_request("QQ", expression))


def test_powered_disjoint_binomial_keeps_per_term_denominators() -> None:
    p = 10**127 + 39
    q = 10**127 + 79
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
                                "value": {"num": 1, "den": p},
                            },
                            "exponent": 4,
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
                                "value": {"num": 1, "den": q},
                            },
                            "exponent": 4,
                        },
                    ],
                },
            ],
        },
        "exponent": 12,
    }
    result = normalize_polynomial_expression(
        _request("QQ", expression, variables=("x", "y"))
    )
    assert len(result.polynomial.polynomial.terms) == 13


def test_multivariate_power_includes_colliding_denominator_mass() -> None:
    """A dependent multivariate support must not use the per-term tight bound.

    ``(sum x^i y^j / d[i,j]^32)^2`` has colliding products, so the ``x^2 y^2``
    coefficient combines every denominator; the uniquely-decomposable shortcut
    must not apply to this linearly dependent exponent-vector support.
    """
    operands = []
    for index, (i, j) in enumerate((i, j) for i in range(3) for j in range(3)):
        denominator = 10**113 + 7 * (index + 1)
        operands.append(
            {
                "kind": "MULTIPLY",
                "operands": [
                    {
                        "kind": "POWER",
                        "base": {"kind": "VARIABLE", "name": "x"},
                        "exponent": i,
                    },
                    {
                        "kind": "POWER",
                        "base": {"kind": "VARIABLE", "name": "y"},
                        "exponent": j,
                    },
                    {
                        "kind": "POWER",
                        "base": {
                            "kind": "LITERAL",
                            "value": {"num": 1, "den": denominator},
                        },
                        "exponent": 32,
                    },
                ],
            }
        )
    expression = {
        "kind": "POWER",
        "base": {"kind": "ADD", "operands": operands},
        "exponent": 2,
    }
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(
            _request("QQ", expression, variables=("x", "y"))
        )


def test_power_of_one_preserves_the_base_aggregate_size() -> None:
    """``POWER(P, 1)`` is the identity and must not inflate the size bound.

    A sparse base with one wide coefficient is admitted directly; wrapping it
    in ``POWER(..., 1)`` must not replace its aggregate size with
    ``support * maximum coefficient width``.
    """

    operands = [
        {
            "kind": "MULTIPLY",
            "operands": [
                {
                    "kind": "POWER",
                    "base": {"kind": "VARIABLE", "name": "x"},
                    "exponent": index,
                },
                {
                    "kind": "LITERAL",
                    "value": {
                        "num": 10**100 + 1 if index == 0 else 1,
                        "den": 1,
                    },
                },
            ],
        }
        for index in range(1, 33)
    ]
    base = {
        "kind": "POWER",
        "base": {"kind": "ADD", "operands": operands},
        "exponent": 1,
    }
    wrapped = {"kind": "POWER", "base": base, "exponent": 1}
    direct_result = normalize_polynomial_expression(_request("ZZ", base))
    wrapped_result = normalize_polynomial_expression(_request("ZZ", wrapped))
    assert wrapped_result.polynomial.polynomial == direct_result.polynomial.polynomial
    from jacobian.math.polynomials._expression_normalize import _metrics

    inner = _request("ZZ", {"kind": "ADD", "operands": operands})
    outer = _request("ZZ", wrapped)
    assert (
        _metrics(outer.expression).total_coefficient_digits
        <= _metrics(inner.expression).total_coefficient_digits
    )


def _powered_sum_square(i: int, exponent: int) -> dict[str, Any]:
    p = 10**127 + 7 * (2 * i)
    q = 10**127 + 7 * (2 * i + 1)
    inner = {
        "kind": "ADD",
        "operands": [
            {
                "kind": "MULTIPLY",
                "operands": [
                    {
                        "kind": "POWER",
                        "base": {"kind": "VARIABLE", "name": "x"},
                        "exponent": i,
                    },
                    {
                        "kind": "POWER",
                        "base": {"kind": "VARIABLE", "name": "y"},
                        "exponent": i + 1,
                    },
                    {
                        "kind": "POWER",
                        "base": {"kind": "LITERAL", "value": {"num": 1, "den": p}},
                        "exponent": exponent,
                    },
                ],
            },
            {
                "kind": "MULTIPLY",
                "operands": [
                    {
                        "kind": "POWER",
                        "base": {"kind": "VARIABLE", "name": "x"},
                        "exponent": 20 - i,
                    },
                    {
                        "kind": "POWER",
                        "base": {"kind": "VARIABLE", "name": "y"},
                        "exponent": 19 - i,
                    },
                    {
                        "kind": "POWER",
                        "base": {"kind": "LITERAL", "value": {"num": 1, "den": q}},
                        "exponent": exponent,
                    },
                ],
            },
        ],
    }
    return {"kind": "POWER", "base": inner, "exponent": 2}


def test_power_support_keys_include_mixed_products() -> None:
    """A multi-term power's support is the Minkowski sum, not the pure powers."""
    expression = {
        "kind": "ADD",
        "operands": [_powered_sum_square(i, 32) for i in range(1, 6)],
    }
    with pytest.raises(OperationResourceAdmissionError):
        normalize_polynomial_expression(
            _request("QQ", expression, variables=("x", "y"))
        )


def test_identity_power_preserves_intermediate_digits() -> None:
    """Wrapping an accepted expression in POWER(..., 1) cannot inflate its size."""
    base = {
        "kind": "ADD",
        "operands": [
            {
                "kind": "POWER",
                "base": {"kind": "LITERAL", "value": {"num": 10**127, "den": 1}},
                "exponent": 32,
            },
            *(
                {
                    "kind": "MULTIPLY",
                    "operands": [
                        {
                            "kind": "ADD",
                            "operands": [
                                {"kind": "LITERAL", "value": {"num": 1, "den": 1}},
                                {"kind": "VARIABLE", "name": f"x{index}"},
                                {
                                    "kind": "POWER",
                                    "base": {"kind": "VARIABLE", "name": f"x{index}"},
                                    "exponent": 2,
                                },
                            ],
                        }
                    ],
                }
                for index in range(1, 8)
            ),
        ],
    }
    variables = tuple(f"x{index}" for index in range(1, 8))
    wrapped = {"kind": "POWER", "base": base, "exponent": 1}
    direct_result = normalize_polynomial_expression(_request("QQ", base, variables))
    wrapped_result = normalize_polynomial_expression(_request("QQ", wrapped, variables))
    assert wrapped_result.polynomial.polynomial == direct_result.polynomial.polynomial
    from jacobian.math.polynomials._expression_normalize import _metrics

    direct_metrics = _metrics(_request("QQ", base, variables).expression)
    wrapped_metrics = _metrics(_request("QQ", wrapped, variables).expression)
    assert wrapped_metrics.intermediate_digits == direct_metrics.intermediate_digits


def test_constant_plus_variable_power_uses_affine_uniqueness() -> None:
    """``(1/p**4 + x/q**4)**12`` is admitted with a 13-term result."""
    p = 10**127 + 3
    q = 10**127 + 7

    def lit(num: int, den: int) -> dict[str, Any]:
        return {"kind": "LITERAL", "value": {"num": num, "den": den}}

    expression = {
        "kind": "POWER",
        "base": {
            "kind": "ADD",
            "operands": [
                {"kind": "POWER", "base": lit(1, p), "exponent": 4},
                {
                    "kind": "MULTIPLY",
                    "operands": [
                        {"kind": "VARIABLE", "name": "x"},
                        {"kind": "POWER", "base": lit(1, q), "exponent": 4},
                    ],
                },
            ],
        },
        "exponent": 12,
    }
    result = normalize_polynomial_expression(_request("QQ", expression))
    assert len(result.polynomial.polynomial.terms) == 13


def test_mutually_exclusive_factor_denominators_are_not_summed() -> None:
    """A product of binomials selects one denominator per factor."""
    variables = tuple(f"a{index}" for index in range(6))
    factors = []
    for index in range(3):
        left = 10**127 + 11 * index
        right = left + 5
        factors.append(
            {
                "kind": "ADD",
                "operands": [
                    {
                        "kind": "MULTIPLY",
                        "operands": [
                            {"kind": "VARIABLE", "name": f"a{2 * index}"},
                            {
                                "kind": "POWER",
                                "base": {
                                    "kind": "LITERAL",
                                    "value": {"num": 1, "den": left},
                                },
                                "exponent": 16,
                            },
                        ],
                    },
                    {
                        "kind": "MULTIPLY",
                        "operands": [
                            {"kind": "VARIABLE", "name": f"a{2 * index + 1}"},
                            {
                                "kind": "POWER",
                                "base": {
                                    "kind": "LITERAL",
                                    "value": {"num": 1, "den": right},
                                },
                                "exponent": 16,
                            },
                        ],
                    },
                ],
            }
        )
    expression = {"kind": "MULTIPLY", "operands": factors}
    result = normalize_polynomial_expression(_request("QQ", expression, variables))
    assert len(result.polynomial.polynomial.terms) == 8


@pytest.mark.parametrize(
    ("factor_count", "denominator_digits"), [(4, 40), (6, 100), (8, 80)]
)
def test_colliding_product_envelope_covers_the_coefficient_it_returns(
    factor_count: int, denominator_digits: int
) -> None:
    """A colliding product must pay for reaching a common denominator.

    Each factor here contributes two pairwise-coprime denominators, so several
    factor choices land on the same exponent and their products are summed over
    the lcm of every denominator in the product. The previous numerator bound
    added only the child numerator heights and a collision count, which
    under-called the returned coefficient by five- to thirteen-fold on these
    accepted requests: the declared coefficient-height envelope was not a
    property of the result actually produced.
    """

    low = 10 ** (denominator_digits - 1)
    seen: set[int] = set()

    def fresh_prime() -> int:
        candidate = int(sympy.randprime(low, 10**denominator_digits - 1))
        while candidate in seen:
            candidate = int(sympy.randprime(low, 10**denominator_digits - 1))
        seen.add(candidate)
        return candidate

    numerator = 2**60
    factors = tuple(
        {
            "kind": "ADD",
            "operands": [
                {"kind": "LITERAL", "value": {"num": numerator, "den": fresh_prime()}},
                {
                    "kind": "MULTIPLY",
                    "operands": [
                        {"kind": "LITERAL", "value": {"num": 1, "den": fresh_prime()}},
                        {"kind": "VARIABLE", "name": "x"},
                    ],
                },
            ],
        }
        for _ in range(factor_count)
    )
    expression: dict[str, Any] = {
        "kind": "MULTIPLY",
        "operands": [factors[0], factors[1]],
    }
    for factor in factors[2:]:
        expression = {"kind": "MULTIPLY", "operands": [expression, factor]}

    request = _request("QQ", expression)
    metrics = _metrics(request.expression)
    result = normalize_polynomial_expression(request)
    returned_bits = max(
        term.coefficient.num.bit_length() for term in result.polynomial.polynomial.terms
    )
    assert returned_bits <= metrics.maximum_numerator_bits
    # Every factor contributes two coprime denominators, so the envelope has to
    # carry that mass: the collision-blind estimate charged only child
    # numerators and a collision count.
    assert metrics.denominator_mass_bits > 0


def test_noncolliding_product_charges_no_denominator_scaling() -> None:
    """Only a colliding product pays for a common denominator.

    When every factor choice lands on its own exponent the numerator bound must
    stay what it was, so the repair does not turn ordinary disjoint products
    into rejected requests.
    """

    factors = (
        {
            "kind": "ADD",
            "operands": [
                {"kind": "LITERAL", "value": {"num": 1, "den": 10**120 + 7}},
                {
                    "kind": "MULTIPLY",
                    "operands": [
                        {"kind": "LITERAL", "value": {"num": 1, "den": 10**120 + 19}},
                        {"kind": "VARIABLE", "name": "a"},
                    ],
                },
            ],
        },
        {
            "kind": "ADD",
            "operands": [
                {"kind": "LITERAL", "value": {"num": 1, "den": 10**120 + 31}},
                {
                    "kind": "MULTIPLY",
                    "operands": [
                        {"kind": "LITERAL", "value": {"num": 1, "den": 10**120 + 37}},
                        {"kind": "VARIABLE", "name": "b"},
                    ],
                },
            ],
        },
    )
    request = _request(
        "QQ", {"kind": "MULTIPLY", "operands": list(factors)}, ("a", "b")
    )
    from jacobian.math.polynomials._expression_normalize import (
        _product_denominator_scaling_bits,
        _product_support_collision,
    )

    metrics = _metrics(request.expression)
    expression = request.expression
    assert isinstance(expression, PolynomialMultiply)
    child_metrics = [_metrics(operand) for operand in expression.operands]
    assert not _product_support_collision(child_metrics)
    assert _product_denominator_scaling_bits(child_metrics, colliding=False) == 0
    assert _product_denominator_scaling_bits(child_metrics, colliding=True) > 0
    # The combined envelope is exactly the collision-blind sum: no denominator
    # mass was charged onto the numerator because no two choices are summed.
    assert metrics.maximum_numerator_bits == sum(
        child.numerator_bits + _ceil_log2(child.support) for child in child_metrics
    )
    result = normalize_polynomial_expression(request)
    assert len(result.polynomial.polynomial.terms) == 4
