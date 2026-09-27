"""Exact arithmetic checks for modular-form cyclotomic coefficient fields."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.cyclotomic import (
    add,
    divide,
    inverse,
    multiply,
    subtract,
)


def _element(order: int, *coefficients: int | Fraction) -> RationalCyclotomicElement:
    field = RationalCyclotomicField(order=order)
    values = tuple(Fraction(value) for value in coefficients)
    values += (Fraction(0),) * (field.degree - len(values))
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(value) for value in values
        ),
    )


def _values(value: RationalCyclotomicElement) -> tuple[Fraction, ...]:
    return tuple(item.as_fraction() for item in value.coefficients_ascending)


def test_cyclotomic_reduction_for_fourth_and_third_roots() -> None:
    z4 = _element(4, 0, 1)
    assert _values(multiply(z4, z4)) == (Fraction(-1), Fraction(0))

    z3 = _element(3, 0, 1)
    one = _element(3, 1)
    assert _values(add(add(multiply(z3, z3), z3), one)) == (Fraction(0), Fraction(0))


def test_inverse_and_division_are_exact() -> None:
    value = _element(5, 2, 1)
    reciprocal = inverse(value)
    assert _values(multiply(value, reciprocal)) == (Fraction(1),) + (Fraction(0),) * 3
    assert divide(multiply(value, value), value) == value


def test_inverse_admits_low_height_element_in_degree_six() -> None:
    value = _element(7, 1, 1)
    assert (
        _values(multiply(value, inverse(value))) == (Fraction(1),) + (Fraction(0),) * 5
    )


def test_inverse_height_bound_rejects_before_cyclotomic_expansion(monkeypatch) -> None:
    value = _element(7, 10_000, 1)

    def unexpected_work(*_args: object) -> tuple[int, ...]:
        pytest.fail("height admission must precede polynomial or matrix construction")

    monkeypatch.setattr(cyclotomic, "_cyclotomic_polynomial", unexpected_work)
    monkeypatch.setattr(cyclotomic, "_solve_coordinates", unexpected_work)
    with pytest.raises(OperationResourceAdmissionError):
        inverse(value)


def test_arithmetic_revalidates_forged_model_construct_inputs() -> None:
    valid = _element(5, 1)
    one = CanonicalRational.from_integer_ratio(1, 1)
    malformed_values = (
        RationalCyclotomicElement.model_construct(
            field=RationalCyclotomicField(order=5),
            coefficients_ascending=(one,),
        ),
        RationalCyclotomicElement.model_construct(
            field=RationalCyclotomicField(order=5),
            coefficients_ascending=(one, "forged", one, one),
        ),
        RationalCyclotomicElement.model_construct(
            field=RationalCyclotomicField.model_construct(
                domain="QQ_CYCLOTOMIC", order=0, generator="CLASS_OF_X"
            ),
            coefficients_ascending=(one, one, one, one),
        ),
        RationalCyclotomicElement.model_construct(
            field=RationalCyclotomicField(order=5),
            coefficients_ascending=(
                CanonicalRational.model_construct(num=10**256, den=1),
                one,
                one,
                one,
            ),
        ),
    )
    for malformed in malformed_values:
        with pytest.raises(
            (OperationDomainValidationError, OperationResourceAdmissionError)
        ):
            add(valid, malformed)
    with pytest.raises(
        (OperationDomainValidationError, OperationResourceAdmissionError)
    ):
        divide(valid, malformed_values[1])


def test_addition_multiplication_and_distributivity() -> None:
    a, b, c = _element(8, 1, 2), _element(8, -3, 1), _element(8, 2, 0, 1)
    assert add(a, b) == subtract(add(add(a, b), c), c)
    assert multiply(a, add(b, c)) == add(multiply(a, b), multiply(a, c))


def test_arithmetic_rejects_parent_mismatch_and_zero_inverse() -> None:
    with pytest.raises(OperationDomainValidationError):
        add(_element(3, 1), _element(4, 1))
    with pytest.raises(OperationDomainValidationError):
        inverse(_element(3, 0))
