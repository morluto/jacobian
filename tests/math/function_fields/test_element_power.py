"""Exact, bounded exponentiation in finite function fields."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields import (
    FiniteFunctionFieldElement,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
    function_field_element_power,
)
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FunctionFieldElementPowerRequest,
)
from jacobian.math.function_fields._tools import TOOLS

OPERATION_ID = "function_field.element.power.compute"


def _rf(
    prime: int, numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=prime, coefficients=numerator),
        denominator=PrimeFieldPolynomial(
            characteristic=prime, coefficients=denominator
        ),
    )


def _rational_field(prime: int = 5) -> FiniteFunctionField:
    return FiniteFunctionField(
        characteristic=prime,
        variable="x",
        generator="y",
        defining_polynomial=(_rf(prime, (1,)),),
    )


def _element(
    field: FiniteFunctionField,
    coordinates: tuple[PrimeFieldRationalFunction, ...],
) -> FiniteFunctionFieldElement:
    return FiniteFunctionFieldElement(field=field, coordinates=coordinates)


def _quadratic_field() -> FiniteFunctionField:
    # y^2 + y + x is separable and irreducible over GF(2)(x).
    return FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="y",
        defining_polynomial=(_rf(2, (0, 1)), _rf(2, (1,)), _rf(2, (1,))),
    )


def test_rational_function_power_has_exact_coefficients_and_parent() -> None:
    field = _rational_field()
    x = _element(field, (_rf(5, (0, 1)),))

    powered = function_field_element_power(x, 5)

    assert powered.field == field
    assert powered.coordinates == (_rf(5, (0, 0, 0, 0, 0, 1)),)


def test_coefficient_degree_boundary_is_accepted_at_twelve() -> None:
    field = _rational_field()
    x = _element(field, (_rf(5, (0, 1)),))

    powered = function_field_element_power(x, 12)

    assert powered.coordinates == (_rf(5, (0,) * 12 + (1,)),)


def test_extension_power_reduces_generator_relation_exactly() -> None:
    field = _quadratic_field()
    y = _element(field, (_rf(2, (0,)), _rf(2, (1,))))

    square = function_field_element_power(y, 2)
    fourth = function_field_element_power(y, 4)

    # In characteristic two, y^2=y+x, hence y^4=y+x+x^2.
    assert square.field == field
    assert square.coordinates == (_rf(2, (0, 1)), _rf(2, (1,)))
    assert fourth.coordinates == (_rf(2, (0, 1, 1)), _rf(2, (1,)))


def test_zero_and_one_exponents_keep_the_exact_parent() -> None:
    field = _quadratic_field()
    zero = _element(field, (_rf(2, (0,)), _rf(2, (0,))))
    y = _element(field, (_rf(2, (0,)), _rf(2, (1,))))

    assert function_field_element_power(zero, 0).coordinates == (
        _rf(2, (1,)),
        _rf(2, (0,)),
    )
    assert function_field_element_power(zero, 0).field == field
    assert function_field_element_power(zero, 1) == zero
    assert function_field_element_power(y, 1) == y


def test_degree_growth_is_rejected_before_coefficient_arithmetic(monkeypatch) -> None:
    import jacobian.math.function_fields.operations as operations

    field = _rational_field()
    x = _element(field, (_rf(5, (0, 1)),))

    def unexpected_product(*args, **kwargs):
        raise AssertionError("power arithmetic ran before admission")

    monkeypatch.setattr(operations, "_power_product", unexpected_product)
    with pytest.raises(OperationResourceAdmissionError, match="degree bound"):
        function_field_element_power(x, 13)


@pytest.mark.parametrize("exponent", [-1, True])
def test_negative_or_noninteger_exponents_are_rejected(exponent: int) -> None:
    x = _element(_rational_field(), (_rf(5, (0, 1)),))

    with pytest.raises(
        OperationDomainValidationError, match="nonnegative strict integer"
    ):
        function_field_element_power(x, exponent)


def test_power_request_round_trips_and_operation_is_published() -> None:
    field = _rational_field()
    request = FunctionFieldElementPowerRequest(
        element=_element(field, (_rf(5, (0, 1)),)), exponent=3
    )
    decoded = FunctionFieldElementPowerRequest.model_validate_json(
        request.model_dump_json()
    )

    assert decoded == request
    assert OPERATION_ID in {tool.operation_id for tool in TOOLS}
