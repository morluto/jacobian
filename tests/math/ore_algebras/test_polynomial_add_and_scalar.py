"""Bounded additive algebra for polynomial-coefficient shift operators."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import (
    ShiftOperatorAddRequest,
    ShiftOreOperator,
)
from jacobian.math.ore_algebras.operations import (
    shift_operator_add,
    shift_operator_multiply,
    shift_operator_normalize_polynomial_coefficients,
    shift_operator_scalar_left_multiply,
)
from jacobian.math.polynomials.values import RationalFunction


def _rf(
    numerator: tuple[tuple[Fraction | int, int], ...],
    denominator: tuple[tuple[Fraction | int, int], ...] = ((1, 0),),
) -> RationalFunction:
    def _part(terms: tuple[tuple[Fraction | int, int], ...]) -> dict[str, object]:
        return {
            "terms": [
                {
                    "coefficient": {
                        "num": Fraction(value).numerator,
                        "den": Fraction(value).denominator,
                    },
                    "exponents": [exponent],
                }
                for value, exponent in sorted(
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


def _op(terms: tuple[tuple[int, RationalFunction], ...]) -> ShiftOreOperator:
    return ShiftOreOperator.model_validate(
        {
            "variable": "n",
            "terms": [
                {"exponent": exponent, "coefficient": coefficient}
                for exponent, coefficient in terms
            ],
        }
    )


def _coefficient(operator: ShiftOreOperator, exponent: int) -> dict[int, Fraction]:
    term = next(term for term in operator.terms if term.exponent == exponent)
    return {
        item.exponents[0]: item.coefficient.as_fraction()
        for item in term.coefficient.numerator.terms
    }


def _direct_sequence_action(
    operator: ShiftOreOperator, values: tuple[int, ...], n: int
) -> Fraction:
    total = Fraction(0)
    for term in operator.terms:
        numerator = sum(
            (
                item.coefficient.as_fraction() * n ** item.exponents[0]
                for item in term.coefficient.numerator.terms
            ),
            Fraction(0),
        )
        sequence_value = Fraction(values[n + term.exponent])
        total += numerator * sequence_value
    return total


def test_addition_sums_same_exponents_and_removes_zero_coefficients() -> None:
    left = _op(((0, _rf(((1, 1), (2, 0)))), (2, _rf(((1, 0),)))))
    right = _op(((0, _rf(((-1, 1), (-2, 0)))), (1, _rf(((3, 0),)))))

    result = shift_operator_add(left, right)

    assert [
        (term.exponent, _coefficient(result.sum, term.exponent))
        for term in result.sum.terms
    ] == [
        (1, {0: Fraction(3)}),
        (2, {0: Fraction(1)}),
    ]


def test_addition_matches_independent_sequence_action_oracle() -> None:
    left = _op(((0, _rf(((Fraction(1, 3), 1), (1, 0)))), (2, _rf(((2, 0),)))))
    right = _op(((0, _rf(((Fraction(2, 5), 1), (-1, 0)))), (1, _rf(((3, 0),)))))
    result = shift_operator_add(left, right)
    values = (2, -1, 4, 3, 5, -2, 7)

    for n in range(len(values) - 2):
        assert _direct_sequence_action(result.sum, values, n) == (
            _direct_sequence_action(left, values, n)
            + _direct_sequence_action(right, values, n)
        )


def test_scalar_left_multiplication_is_exact_over_rational_constants() -> None:
    source = _op(((0, _rf(((Fraction(1, 3), 1), (2, 0)))), (2, _rf(((1, 0),)))))
    scalar = _rf(((Fraction(-5, 7), 0),))

    result = shift_operator_scalar_left_multiply(scalar, source)

    assert result.product.terms[0].coefficient.numerator.terms[
        0
    ].coefficient.as_fraction() == Fraction(-5, 21)
    values = (3, 1, 5, -2, 4, 8)
    for n in range(len(values) - 2):
        assert _direct_sequence_action(result.product, values, n) == (
            Fraction(-5, 7) * _direct_sequence_action(source, values, n)
        )


def test_ore_commutation_relation_composes_with_addition() -> None:
    shift = _op(((1, _rf(((1, 0),))),))
    coordinate = _op(((0, _rf(((1, 1),))),))
    one_shift = shift
    n_times_shift = _op(((1, _rf(((1, 1),))),))

    left_product = shift_operator_multiply(shift, coordinate).product
    right_sum = shift_operator_add(n_times_shift, one_shift).sum

    assert left_product == right_sum
    values = (3, 4, -1, 6, 2)
    for index in range(len(values) - 1):
        expected = (index + 1) * values[index + 1]
        assert _direct_sequence_action(left_product, values, index) == expected
        assert _direct_sequence_action(right_sum, values, index) == expected


def test_rational_function_operator_coefficients_are_outside_addition_subring() -> None:
    reciprocal = _rf(((1, 0),), ((1, 1),))
    operator = _op(((0, reciprocal),))
    with pytest.raises(OperationDomainValidationError, match="polynomial coefficients"):
        shift_operator_add(operator, _op(()))

    with pytest.raises(ValueError, match="polynomial coefficients"):
        ShiftOperatorAddRequest(left=operator, right=_op(()))


def test_nonconstant_scalar_is_not_mistaken_for_a_rational_constant() -> None:
    with pytest.raises(OperationDomainValidationError, match="rational constant"):
        shift_operator_scalar_left_multiply(_rf(((1, 1),)), _op(((0, _rf(((1, 0),))),)))


def test_normalization_returns_exact_source_scale_and_primitive_operator() -> None:
    source = _op(
        (
            (0, _rf(((Fraction(4, 3), 0),))),
            (1, _rf(((Fraction(2, 3), 1),))),
        )
    )

    result = shift_operator_normalize_polynomial_coefficients(source)

    assert result.scale.variables == ("n",)
    assert result.scale.denominator.terms[0].coefficient.as_fraction() == 1
    assert result.scale.numerator.terms[0].coefficient.as_fraction() == Fraction(2, 3)
    assert _coefficient(result.normalized, 0) == {0: Fraction(2)}
    assert _coefficient(result.normalized, 1) == {1: Fraction(1)}
    rebuilt = shift_operator_scalar_left_multiply(result.scale, result.normalized)
    assert rebuilt.product == source


def test_normalization_sign_makes_leading_operator_coefficient_positive() -> None:
    source = _op(
        (
            (0, _rf(((Fraction(-3, 2), 0),))),
            (1, _rf(((Fraction(-1, 2), 1),))),
        )
    )

    result = shift_operator_normalize_polynomial_coefficients(source)

    assert result.scale.variables == ("n",)
    assert result.scale.denominator.terms[0].coefficient.as_fraction() == 1
    assert result.scale.numerator.terms[0].coefficient.as_fraction() == Fraction(-1, 2)
    assert _coefficient(result.normalized, 0) == {0: Fraction(3)}
    assert _coefficient(result.normalized, 1) == {1: Fraction(1)}


def test_zero_operator_normalizes_to_itself_with_unit_scale() -> None:
    source = _op(())

    result = shift_operator_normalize_polynomial_coefficients(source)

    assert result.normalized == source
    assert result.scale.variables == ("n",)
    assert result.scale.denominator.terms[0].coefficient.as_fraction() == 1
    assert result.scale.numerator.terms[0].coefficient.as_fraction() == 1


def test_normalization_lcm_growth_is_rejected_within_fixed_intermediate_cap() -> None:
    source = _op(
        (
            (0, _rf(((Fraction(1, 10**63 - 1), 0),))),
            (1, _rf(((Fraction(1, 10**63 - 2), 0),))),
            (2, _rf(((Fraction(1, 10**63 - 3), 0),))),
        )
    )

    with pytest.raises(OperationResourceAdmissionError, match="scale exceeds"):
        shift_operator_normalize_polynomial_coefficients(source)


def test_addition_digit_admission_rejects_before_rational_sum(monkeypatch) -> None:
    from jacobian.math.ore_algebras import operations

    large_a = Fraction(10**64 - 3, 10**64 - 1)
    large_b = Fraction(10**64 - 5, 10**64 - 7)
    left = _op(((0, _rf(((large_a, 0),))),))
    right = _op(((0, _rf(((large_b, 0),))),))

    def fail(*args, **kwargs):
        raise AssertionError("rational addition must follow result-height admission")

    monkeypatch.setattr(operations, "_poly_add", fail)
    with pytest.raises(OperationResourceAdmissionError, match="coefficient-digit"):
        shift_operator_add(left, right)
