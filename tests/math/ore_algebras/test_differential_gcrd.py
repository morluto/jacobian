"""Exact order-one GCRD fixtures and ambient Ore identities."""

from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import DifferentialOreOperator
from jacobian.math.ore_algebras.differential_gcrd.operations import (
    differential_operator_gcrd,
)
from jacobian.math.ore_algebras.operations import differential_operator_multiply
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _constant(value: Fraction | int) -> RationalFunction:
    scalar = CanonicalRational.from_fraction(Fraction(value))
    return RationalFunction(
        domain="QQ",
        variables=("x",),
        numerator=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=scalar, exponents=(0,)),)
        ),
        denominator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(1)),
                    exponents=(0,),
                ),
            )
        ),
    )


def _operator(coefficients: dict[int, Fraction | int]) -> DifferentialOreOperator:
    return DifferentialOreOperator(
        variable="x",
        terms=tuple(
            {"order": order, "coefficient": _constant(value)}
            for order, value in sorted(coefficients.items())
            if value
        ),
    )


def _coefficients(operator: DifferentialOreOperator) -> dict[int, Fraction]:
    result: dict[int, Fraction] = {}
    for term in operator.terms:
        result[term.order] = term.coefficient.numerator.terms[
            0
        ].coefficient.as_fraction()
    return result


def _add(left: DifferentialOreOperator, right: DifferentialOreOperator):
    coefficients = _coefficients(left)
    for order, value in _coefficients(right).items():
        coefficients[order] = coefficients.get(order, Fraction(0)) + value
    return _operator(coefficients)


def _check_relations(result) -> None:
    assert (
        differential_operator_multiply(result.left_cofactor, result.divisor).product
        == result.left
    )
    assert (
        differential_operator_multiply(result.right_cofactor, result.divisor).product
        == result.right
    )
    left_bezout = differential_operator_multiply(
        result.bezout_left, result.left
    ).product
    right_bezout = differential_operator_multiply(
        result.bezout_right, result.right
    ).product
    assert _add(left_bezout, right_bezout) == result.divisor


def test_coprime_first_order_operators_have_exact_bezout_identity() -> None:
    result = differential_operator_gcrd(
        _operator({1: 1, 0: 1}), _operator({1: 1, 0: -1})
    )

    assert result.divisor == _operator({0: 1})
    assert result.bezout_left == _operator({0: Fraction(1, 2)})
    assert result.bezout_right == _operator({0: Fraction(-1, 2)})
    _check_relations(result)


def test_shared_first_order_right_divisor_is_monic_and_maximal() -> None:
    left = _operator({1: 2, 0: 6})
    right = _operator({1: -3, 0: -9})
    result = differential_operator_gcrd(left, right)

    assert result.divisor == _operator({1: 1, 0: 3})
    assert result.left_cofactor == _operator({0: 2})
    assert result.right_cofactor == _operator({0: -3})
    _check_relations(result)


def test_order_zero_and_degenerate_inputs_have_field_unit_semantics() -> None:
    cases = (
        (_operator({}), _operator({})),
        (_operator({}), _operator({1: 2, 0: 4})),
        (_operator({1: 2, 0: 4}), _operator({})),
        (_operator({0: 3}), _operator({0: -5})),
        (_operator({0: 3}), _operator({1: 2, 0: 4})),
    )
    for left, right in cases:
        result = differential_operator_gcrd(left, right)
        _check_relations(result)
        if left.terms == right.terms == ():
            assert result.divisor.terms == ()
        else:
            assert result.divisor == _operator({0: 1}) or result.divisor.order == 1


def test_small_rational_constant_family_replays_exact_gcd_and_bezout() -> None:
    operators = [_operator({})]
    scalars = (Fraction(-2), Fraction(-1, 2), Fraction(1), Fraction(3, 2))
    operators.extend(_operator({0: scalar}) for scalar in scalars)
    operators.extend(
        _operator({1: leading, 0: constant})
        for leading, constant in product(
            (Fraction(-2), Fraction(1), Fraction(3, 2)), scalars
        )
    )

    for left, right in product(operators, repeat=2):
        result = differential_operator_gcrd(left, right)
        _check_relations(result)
        if left.order == right.order == 1:
            a, b = _coefficients(left)[1], _coefficients(left)[0]
            c, d = _coefficients(right)[1], _coefficients(right)[0]
            assert result.divisor.order == (1 if b * c == a * d else 0)


def test_nonconstant_or_higher_order_inputs_are_outside_the_slice() -> None:
    x = RationalFunction(
        domain="QQ",
        variables=("x",),
        numerator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(1)),
                    exponents=(1,),
                ),
            )
        ),
        denominator=_constant(1).denominator,
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="rational constant coefficients only",
    ):
        differential_operator_gcrd(
            DifferentialOreOperator(
                variable="x",
                terms=({"order": 0, "coefficient": x},),
            ),
            _operator({1: 1}),
        )

    with pytest.raises(OperationResourceAdmissionError, match="order at most one"):
        differential_operator_gcrd(_operator({2: 1}), _operator({1: 1}))


def test_large_inputs_with_small_exact_outputs_are_accepted() -> None:
    result = differential_operator_gcrd(
        _operator({1: 1, 0: 10**20}),
        _operator({1: 1, 0: 10**20 + 1}),
    )
    assert result.divisor == _operator({0: 1})
    _check_relations(result)


def test_twenty_digit_inputs_remain_composable_with_bounded_outputs() -> None:
    result = differential_operator_gcrd(
        _operator({1: 1, 0: 10**19}), _operator({1: 1, 0: -(10**19)})
    )

    assert result.divisor == _operator({0: 1})
    _check_relations(result)
    assert all(
        max(
            len(str(abs(term.coefficient.numerator.terms[0].coefficient.num))),
            len(str(term.coefficient.numerator.terms[0].coefficient.den)),
        )
        <= 64
        for operator in (
            result.divisor,
            result.left_cofactor,
            result.right_cofactor,
            result.bezout_left,
            result.bezout_right,
        )
        for term in operator.terms
    )
