"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/function_fields/test_function_fields.py``. The preamble is carried over verbatim so the moved tests
resolve every name they use.
"""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import (
    function_field_element_multiply,
)

OPERATION_ID = "function_field.element.multiply.compute"


def _poly(characteristic: int, coefficients: tuple[int, ...]) -> PrimeFieldPolynomial:
    return PrimeFieldPolynomial(
        characteristic=characteristic, coefficients=coefficients
    )


def _rf(
    characteristic: int, numerator: tuple[int, ...], denominator: tuple[int, ...]
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=_poly(characteristic, numerator),
        denominator=_poly(characteristic, denominator),
    )


def _field(
    characteristic: int, defining: tuple[PrimeFieldRationalFunction, ...]
) -> FiniteFunctionField:
    return FiniteFunctionField(
        characteristic=characteristic,
        variable="x",
        generator="y",
        defining_polynomial=defining,
    )


def _element(
    field: FiniteFunctionField, coordinates: tuple[PrimeFieldRationalFunction, ...]
) -> FiniteFunctionFieldElement:
    return FiniteFunctionFieldElement(field=field, coordinates=coordinates)


def _zero(characteristic: int) -> PrimeFieldRationalFunction:
    return _rf(characteristic, (0,), (1,))


def _one(characteristic: int) -> PrimeFieldRationalFunction:
    return _rf(characteristic, (1,), (1,))


def _x(characteristic: int) -> PrimeFieldRationalFunction:
    return _rf(characteristic, (0, 1), (1,))


def _multiply(
    left: FiniteFunctionFieldElement, right: FiniteFunctionFieldElement
) -> FiniteFunctionFieldElement:
    return function_field_element_multiply(left, right).product


def _equal(left: FiniteFunctionFieldElement, right: FiniteFunctionFieldElement) -> bool:
    return left.model_dump_json() == right.model_dump_json()


def _polynomial_order(coefficients: tuple[int, ...], characteristic: int) -> int | None:
    for index, value in enumerate(coefficients):
        if value % characteristic:
            return index
    return None


def _valuation(value: PrimeFieldRationalFunction, characteristic: int) -> int:
    numerator = _polynomial_order(value.numerator.coefficients, characteristic)
    denominator = _polynomial_order(value.denominator.coefficients, characteristic)
    assert numerator is not None and denominator is not None
    return numerator - denominator


def _poly_expression(coefficients: tuple[int, ...], x_symbol) -> object:
    if coefficients == ():
        return 0
    return sum(
        coefficient * x_symbol**power for power, coefficient in enumerate(coefficients)
    )


def test_catalog_discovery() -> None:
    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in ids
    assert "number_theory.function_field.multiply.compute" not in ids


GF2_FIELD = _field(
    2,
    (
        _x(2),
        _one(2),
        _one(2),
    ),
)
GF2_Y = _element(GF2_FIELD, (_zero(2), _one(2)))
GF2_ONE = _element(GF2_FIELD, (_one(2), _zero(2)))
GF3_FIELD = _field(
    3,
    (
        _rf(3, (0, 2), (1,)),  # -x
        _zero(3),
        _one(3),
    ),
)
GF3_Y = _element(GF3_FIELD, (_zero(3), _one(3)))
GF3_FALLBACK_FIELD = _field(
    3,
    (
        _rf(3, (0, 1, 0, 2), (1, 0, 1)),
        _zero(3),
        _one(3),
    ),
)
GF3_FALLBACK_Y = _element(GF3_FALLBACK_FIELD, (_zero(3), _one(3)))
