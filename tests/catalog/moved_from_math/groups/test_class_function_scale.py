"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/groups/test_class_function_scale.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.groups.characters._models import (
    CyclotomicValue,
)


def _value(order: int, coefficients: tuple[int, ...]) -> CyclotomicValue:
    return CyclotomicValue(
        order=order,
        coefficients=tuple(
            CanonicalRational.from_fraction(Fraction(value)) for value in coefficients
        ),
    )


def test_catalog_scale_declaration_executes_example() -> None:
    tool = next(
        candidate
        for candidate in BUILTIN_TOOLS
        if candidate.operation_id == "class_function.scale.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert tuple(value.coefficients[0].as_fraction() for value in result.values) == (
        Fraction(-4),
        Fraction(0),
        Fraction(2),
    )
