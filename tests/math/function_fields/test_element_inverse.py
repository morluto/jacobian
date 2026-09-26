"""Exact inversion over bounded finite function fields."""

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
    function_field_element_inverse,
    function_field_element_multiply,
)
from jacobian.math.function_fields._models import FiniteFunctionField
from jacobian.math.function_fields._tools import TOOLS

OPERATION_ID = "function_field.element.inverse.compute"


def _rf(numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)):
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=2, coefficients=numerator),
        denominator=PrimeFieldPolynomial(characteristic=2, coefficients=denominator),
    )


def _field() -> FiniteFunctionField:
    # y^2 + y + x is irreducible and separable over GF(2)(x).
    return FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="y",
        defining_polynomial=(_rf((0, 1)), _rf((1,)), _rf((1,))),
    )


def _element(coordinates: tuple[PrimeFieldRationalFunction, ...]):
    return FiniteFunctionFieldElement(field=_field(), coordinates=coordinates)


def test_generator_inverse_multiplies_to_the_parent_unit() -> None:
    generator = _element((_rf((0,)), _rf((1,))))
    inverse = function_field_element_inverse(generator)

    assert inverse.field == generator.field
    assert function_field_element_multiply(generator, inverse).product.coordinates == (
        _rf((1,)),
        _rf((0,)),
    )


def test_rational_function_inverse_is_exact_and_parent_bound() -> None:
    field = _field()
    value = FiniteFunctionFieldElement(
        field=field, coordinates=(_rf((1, 1)), _rf((0,)))
    )

    inverse = function_field_element_inverse(value)

    assert inverse.field == field
    assert inverse.coordinates == (_rf((1,), (1, 1)), _rf((0,)))


def test_zero_has_no_inverse() -> None:
    zero = _element((_rf((0,)), _rf((0,))))

    with pytest.raises(OperationDomainValidationError, match="zero element"):
        function_field_element_inverse(zero)


def test_reducible_parent_is_a_domain_error() -> None:
    field = FiniteFunctionField(
        characteristic=2,
        variable="x",
        generator="y",
        defining_polynomial=(_rf((0,)), _rf((1,)), _rf((1,))),
    )
    value = FiniteFunctionFieldElement(field=field, coordinates=(_rf((1,)), _rf((0,))))

    with pytest.raises(OperationDomainValidationError, match="reducible"):
        function_field_element_inverse(value)


def test_conservative_coefficient_growth_boundary_rejects_before_inversion() -> None:
    value = _element((_rf((1, 0, 0, 1)), _rf((1,))))

    with pytest.raises(OperationResourceAdmissionError, match="coefficient bound"):
        function_field_element_inverse(value)


def test_inverse_is_published_in_owner_manifest() -> None:
    assert OPERATION_ID in {tool.operation_id for tool in TOOLS}
