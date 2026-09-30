"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/function_fields/test_element_addition.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.function_fields import (
    FiniteFunctionFieldElement,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
    function_field_element_add,
)
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FunctionFieldElementAddRequest,
)
from jacobian.math.function_fields._tools import TOOLS

OPERATION_ID = "function_field.element.add.compute"


def _polynomial(prime: int, coefficients: tuple[int, ...]) -> PrimeFieldPolynomial:
    return PrimeFieldPolynomial(characteristic=prime, coefficients=coefficients)


def _rational(
    prime: int, numerator: tuple[int, ...], denominator: tuple[int, ...] = (1,)
) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=_polynomial(prime, numerator),
        denominator=_polynomial(prime, denominator),
    )


def _rational_field(prime: int) -> FiniteFunctionField:
    return FiniteFunctionField(
        characteristic=prime,
        variable="x",
        generator="y",
        defining_polynomial=(_rational(prime, (1,)),),
    )


def _quadratic_field(prime: int) -> FiniteFunctionField:
    # y^2 - x is irreducible over GF(p)(x), since x has odd valuation at 0.
    return FiniteFunctionField(
        characteristic=prime,
        variable="x",
        generator="y",
        defining_polynomial=(
            _rational(prime, (0, prime - 1)),
            _rational(prime, (0,)),
            _rational(prime, (1,)),
        ),
    )


def _element(
    field: FiniteFunctionField,
    coordinates: tuple[PrimeFieldRationalFunction, ...],
) -> FiniteFunctionFieldElement:
    return FiniteFunctionFieldElement(field=field, coordinates=coordinates)


def _evaluate_polynomial(coefficients: tuple[int, ...], point: int, prime: int) -> int:
    value = 0
    for coefficient in reversed(coefficients):
        value = (value * point + coefficient) % prime
    return value


def _evaluate_rational_function(value: PrimeFieldRationalFunction, point: int) -> int:
    prime = value.characteristic
    numerator = _evaluate_polynomial(value.numerator.coefficients, point, prime)
    denominator = _evaluate_polynomial(value.denominator.coefficients, point, prime)
    assert denominator != 0
    return numerator * pow(denominator, prime - 2, prime) % prime


def _evaluate_quadratic_element(
    value: FiniteFunctionFieldElement, point: int, generator_value: int
) -> int:
    prime = value.field.characteristic
    return (
        sum(
            _evaluate_rational_function(coordinate, point)
            * pow(generator_value, power, prime)
            for power, coordinate in enumerate(value.coordinates)
        )
        % prime
    )


def test_serialization_catalog_and_native_exports() -> None:
    import jacobian.math as math_api

    field = _rational_field(7)
    left = _element(field, (_rational(7, (1,)),))
    right = _element(field, (_rational(7, (2,)),))
    direct = function_field_element_add(left, right)
    restored = FiniteFunctionFieldElement.model_validate_json(direct.model_dump_json())
    assert restored == direct
    assert math_api.function_fields.function_field_element_add(left, right) == direct

    tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
    request = FunctionFieldElementAddRequest(left=left, right=right)
    assert tool.run(request) == direct
    example = next(example for example in tool.examples)
    parsed = tool.request_type.model_validate_json(
        encode_strict_json(example.input), strict=True
    )
    example_result = tool.run(parsed)
    assert example_result.coordinates[0] == _rational(2, (0, 1))
    assert any(candidate.operation_id == OPERATION_ID for candidate in BUILTIN_TOOLS)
