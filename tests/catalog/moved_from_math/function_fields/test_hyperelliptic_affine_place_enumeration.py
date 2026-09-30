"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/function_fields/test_hyperelliptic_affine_place_enumeration.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields._tools import TOOLS
from jacobian.math.function_fields.hyperelliptic_affine_places import (
    HyperellipticAffinePlacesRequest,
    enumerate_hyperelliptic_affine_places,
)


def _rf(coefficients: tuple[int, ...]) -> PrimeFieldRationalFunction:
    return PrimeFieldRationalFunction(
        numerator=PrimeFieldPolynomial(characteristic=5, coefficients=coefficients),
        denominator=PrimeFieldPolynomial(characteristic=5, coefficients=(1,)),
    )


def _field() -> FiniteFunctionField:
    # y^2 = x^3-x over GF(5), with three branch x-coordinates.
    return FiniteFunctionField(
        characteristic=5,
        variable="x",
        generator="y",
        defining_polynomial=(_rf((0, 1, 0, 4)), _rf((0,)), _rf((1,))),
    )


def test_tool_is_published_and_advertised_example_is_valid_json():
    operation_id = "function_field.hyperelliptic_affine_places.enumerate"
    tool = next(tool for tool in TOOLS if tool.operation_id == operation_id)
    assert tool in BUILTIN_TOOLS
    example = tool.examples[0]
    parsed = HyperellipticAffinePlacesRequest.model_validate(example.input)
    assert enumerate_hyperelliptic_affine_places(parsed.field).places
    assert encode_strict_json(example.input)
