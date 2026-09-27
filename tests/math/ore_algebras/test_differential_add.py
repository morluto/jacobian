import json
from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.ore_algebras._models import (
    DifferentialOperatorAddRequest,
    DifferentialOreOperator,
)
from jacobian.math.ore_algebras._tools import TOOLS
from jacobian.math.ore_algebras.operations import (
    differential_operator_add,
    differential_operator_apply,
)
from jacobian.math.polynomials.values import RationalFunction


def _rf(
    numerator: tuple[tuple[int | Fraction, int], ...],
    denominator: tuple[tuple[int, int], ...] = ((1, 0),),
) -> RationalFunction:
    def _part(terms: tuple[tuple[int | Fraction, int], ...]) -> dict[str, object]:
        return {
            "terms": [
                {
                    "coefficient": {
                        "num": Fraction(coefficient).numerator,
                        "den": Fraction(coefficient).denominator,
                    },
                    "exponents": [exponent],
                }
                for coefficient, exponent in sorted(
                    terms, key=lambda term: term[1], reverse=True
                )
            ]
        }

    return RationalFunction.model_validate(
        {
            "domain": "QQ",
            "variables": ["x"],
            "numerator": _part(numerator),
            "denominator": _part(denominator),
        }
    )


def _operator(
    terms: tuple[tuple[int, RationalFunction], ...],
) -> DifferentialOreOperator:
    return DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {"order": order, "coefficient": coefficient}
                for order, coefficient in terms
            ],
        }
    )


def _poly_eval(terms: tuple[tuple[int, Fraction], ...], at: Fraction) -> Fraction:
    return sum(
        (coefficient * at**exponent for exponent, coefficient in terms), Fraction(0)
    )


def _rf_eval(value: RationalFunction, at: Fraction) -> Fraction:
    numerator = tuple(
        (term.exponents[0], term.coefficient.as_fraction())
        for term in value.numerator.terms
    )
    denominator = tuple(
        (term.exponents[0], term.coefficient.as_fraction())
        for term in value.denominator.terms
    )
    return _poly_eval(numerator, at) / _poly_eval(denominator, at)


def _poly_derivative(terms: tuple[tuple[int, Fraction], ...]):
    return tuple(
        (exponent - 1, coefficient * exponent)
        for exponent, coefficient in terms
        if exponent
    )


def _poly_mul(left, right):
    result: dict[int, Fraction] = {}
    for left_exp, left_value in left:
        for right_exp, right_value in right:
            exponent = left_exp + right_exp
            result[exponent] = (
                result.get(exponent, Fraction(0)) + left_value * right_value
            )
    return tuple((exponent, value) for exponent, value in result.items() if value)


def _poly_sub(left, right):
    result = dict(left)
    for exponent, value in right:
        result[exponent] = result.get(exponent, Fraction(0)) - value
    return tuple((exponent, value) for exponent, value in result.items() if value)


def _direct_operator_action_at(
    operator: DifferentialOreOperator, function: RationalFunction, at: Fraction
) -> Fraction:
    numerator = tuple(
        (term.exponents[0], term.coefficient.as_fraction())
        for term in function.numerator.terms
    )
    denominator = tuple(
        (term.exponents[0], term.coefficient.as_fraction())
        for term in function.denominator.terms
    )
    derivative_numerator = numerator
    derivative_denominator = denominator
    total = Fraction(0)
    for derivative_order in range(
        max((term.order for term in operator.terms), default=-1) + 1
    ):
        if derivative_order:
            current_numerator = derivative_numerator
            current_denominator = derivative_denominator
            derivative_numerator, derivative_denominator = (
                _poly_sub(
                    _poly_mul(_poly_derivative(current_numerator), current_denominator),
                    _poly_mul(current_numerator, _poly_derivative(current_denominator)),
                ),
                _poly_mul(current_denominator, current_denominator),
            )
        term = next(
            (term for term in operator.terms if term.order == derivative_order), None
        )
        if term is not None:
            coefficient_value = _rf_eval(term.coefficient, at)
            derivative_value = _poly_eval(derivative_numerator, at) / _poly_eval(
                derivative_denominator, at
            )
            total += coefficient_value * derivative_value
    return total


def test_differential_sum_matches_independent_action_and_serializes() -> None:
    left = _operator(((0, _rf(((1, 0),), ((1, 1), (1, 0)))), (1, _rf(((1, 1),)))))
    right = _operator(((0, _rf(((1, 0),), ((1, 2), (1, 0)))), (2, _rf(((2, 0),)))))
    function = _rf(((1, 0),), ((1, 2), (1, 0)))

    result = differential_operator_add(left, right)

    assert [term.order for term in result.sum.terms] == [0, 1, 2]
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    for at in (Fraction(1), Fraction(2), Fraction(3)):
        expected = _direct_operator_action_at(
            left, function, at
        ) + _direct_operator_action_at(right, function, at)
        applied = differential_operator_apply(result.sum, function).result
        assert _rf_eval(applied, at) == expected


def test_differential_sum_cancels_equal_orders_and_keeps_canonical_zero() -> None:
    coefficient = _rf(((1, 0),), ((1, 1), (1, 0)))
    left = _operator(((0, coefficient), (2, _rf(((3, 0),)))))
    right = _operator(((0, _rf(((-1, 0),), ((1, 1), (1, 0)))), (1, _rf(((1, 0),)))))

    result = differential_operator_add(left, right).sum

    assert [term.order for term in result.terms] == [1, 2]
    zero = differential_operator_add(_operator(()), _operator(())).sum
    assert zero.terms == ()


def test_differential_addition_accepts_degree_128_output_boundary() -> None:
    left_coefficient = _rf(((1, 0),), ((1, 64), (1, 0)))
    right_coefficient = _rf(((1, 0),), ((1, 64), (2, 0)))

    result = differential_operator_add(
        _operator(((0, left_coefficient),)), _operator(((0, right_coefficient),))
    ).sum

    assert result.terms[0].coefficient.denominator.terms[0].exponents == (128,)


def test_differential_addition_rejects_digit_growth_before_expansion() -> None:
    numerator = 10**64 - 1
    first = _rf(((Fraction(numerator, 10**64 - 2), 0),))
    second = _rf(((Fraction(numerator, 10**64 - 3), 0),))

    with pytest.raises(OperationResourceAdmissionError):
        differential_operator_add(_operator(((0, first),)), _operator(((0, second),)))


def test_differential_addition_has_typed_request_and_catalog_example() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "ore.differential.operator.add.compute"
    )
    example = operation.examples[0]
    request = DifferentialOperatorAddRequest.model_validate_json(
        json.dumps(example.input)
    )

    result = operation.run(request)

    assert [term.order for term in result.sum.terms] == [0, 1]
