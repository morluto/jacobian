from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras import operations as ore_operations
from jacobian.math.ore_algebras._models import ShiftOreOperator
from jacobian.math.ore_algebras.operations import (
    shift_operator_apply_to_sequence_prefix,
    shift_operator_power,
)
from jacobian.math.polynomials.values import RationalFunction


def _rf(value: int) -> RationalFunction:
    return _rf_polynomial(((value, 0),))


def _rf_polynomial(terms: tuple[tuple[int, int], ...]) -> RationalFunction:
    return RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["n"],
            "numerator": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": 1},
                        "exponents": [degree],
                    }
                    for coefficient, degree in sorted(
                        terms, key=lambda term: term[1], reverse=True
                    )
                    if coefficient
                ]
            },
            "denominator": {
                "terms": [{"coefficient": {"num": 1, "den": 1}, "exponents": [0]}]
            },
        }
    )


def _rational_rf(
    numerator: tuple[tuple[int, int], ...], denominator: tuple[tuple[int, int], ...]
) -> RationalFunction:
    def _part(terms: tuple[tuple[int, int], ...]) -> dict:
        return {
            "terms": [
                {"coefficient": {"num": coefficient, "den": 1}, "exponents": [degree]}
                for coefficient, degree in sorted(
                    terms, key=lambda term: term[1], reverse=True
                )
            ]
        }

    return RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["n"],
            "numerator": _part(numerator),
            "denominator": _part(denominator),
        }
    )


def _operator(*terms: tuple[int, int]) -> ShiftOreOperator:
    return ShiftOreOperator.model_validate(
        {
            "variable": "n",
            "terms": [
                {"exponent": exponent, "coefficient": _rf(coefficient)}
                for exponent, coefficient in terms
            ],
        }
    )


def test_binomial_power_matches_independent_dictionary_oracle_and_prefix_action() -> (
    None
):
    base = _operator((0, 1), (1, 1))
    result = shift_operator_power(base, 3)

    oracle: dict[int, Fraction] = {0: Fraction(1)}
    for _ in range(3):
        updated: dict[int, Fraction] = {}
        for exponent, coefficient in oracle.items():
            updated[exponent] = updated.get(exponent, Fraction(0)) + coefficient
            updated[exponent + 1] = updated.get(exponent + 1, Fraction(0)) + coefficient
        oracle = {exponent: value for exponent, value in updated.items() if value}

    observed = {
        term.exponent: term.coefficient.numerator.terms[0].coefficient.as_fraction()
        for term in result.power.terms
    }
    assert observed == oracle == {0: 1, 1: 3, 2: 3, 3: 1}

    geometric = FiniteRationalSequence(values=(1, 2, 4, 8, 16, 32))
    action = shift_operator_apply_to_sequence_prefix(result.power, 0, geometric)
    assert [row.residual.as_fraction() for row in action.residuals] == [
        Fraction(27 * 2**index) for index in range(3)
    ]


def test_zero_power_is_identity_and_order_growth_is_rejected_before_products() -> None:
    shift = _operator((1, 1))
    assert shift_operator_power(shift, 0).power == _operator((0, 1))
    with pytest.raises(OperationDomainValidationError):
        shift_operator_power(shift, 17)
    with pytest.raises(OperationResourceAdmissionError):
        shift_operator_power(_operator((2, 1)), 9)


def test_variable_coefficient_power_preflights_and_preserves_shift_commutation() -> (
    None
):
    base = ShiftOreOperator.model_validate(
        {
            "variable": "n",
            "terms": [
                {"exponent": 0, "coefficient": _rf_polynomial(((1, 1),))},
                {"exponent": 1, "coefficient": _rf(1)},
            ],
        }
    )

    square = shift_operator_power(base, 2).power
    oracle = {
        term.exponent: {
            poly_term.exponents[0]: poly_term.coefficient.as_fraction()
            for poly_term in term.coefficient.numerator.terms
        }
        for term in square.terms
    }
    assert oracle == {
        0: {2: Fraction(1)},
        1: {1: Fraction(2), 0: Fraction(1)},
        2: {0: Fraction(1)},
    }


def test_rational_function_coefficients_require_a_proved_whole_power_envelope() -> None:
    base = ShiftOreOperator.model_validate(
        {
            "variable": "n",
            "terms": [
                {"exponent": 0, "coefficient": _rational_rf(((1, 0),), ((1, 1),))},
                {"exponent": 1, "coefficient": _rf(1)},
            ],
        }
    )

    # (1/n + S)^2 is an exact, finite QQ(n)<S> operator, but its rational
    # function normalization growth is not covered by the current whole-power
    # admission recurrence. It must be refused before the first product.
    with pytest.raises(OperationResourceAdmissionError) as error:
        shift_operator_power(base, 2)
    assert (
        error.value.errors()[0]["type"]
        == "ore_algebra.shift_power_coefficient_envelope"
    )


def test_later_degree_rejection_precedes_any_product_expansion(monkeypatch) -> None:
    base = ShiftOreOperator.model_validate(
        {
            "variable": "n",
            "terms": [
                {"exponent": 0, "coefficient": _rf_polynomial(((1, 32),))},
                {"exponent": 1, "coefficient": _rf(1)},
            ],
        }
    )

    def forbidden_multiply(*_args, **_kwargs):
        raise AssertionError("power admission must finish before multiplication")

    monkeypatch.setattr(ore_operations, "shift_operator_multiply", forbidden_multiply)
    with pytest.raises(OperationResourceAdmissionError):
        shift_operator_power(base, 3)
