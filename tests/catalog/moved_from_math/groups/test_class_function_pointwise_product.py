"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/groups/test_class_function_pointwise_product.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.groups.characters._models import (
    ClassAxis,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import (
    class_function_add,
)

SIZES = (1, 3, 2)


def _value(order: int, coefficients: tuple[int | Fraction, ...]) -> CyclotomicValue:
    return CyclotomicValue(
        order=order,
        coefficients=tuple(
            CanonicalRational.from_fraction(Fraction(value)) for value in coefficients
        ),
    )


def _function(
    sizes: tuple[int, ...], values: tuple[CyclotomicValue, ...], order: int = 1
) -> FiniteClassFunction:
    return FiniteClassFunction(
        axis=ClassAxis(
            class_sizes=sizes, group_order=sum(sizes), cyclotomic_order=order
        ),
        values=values,
    )


def _forged(
    values: tuple[CyclotomicValue, ...], sizes: tuple[int, ...] = SIZES, order: int = 1
) -> FiniteClassFunction:
    return FiniteClassFunction.model_construct(
        axis=ClassAxis(
            class_sizes=sizes, group_order=sum(sizes), cyclotomic_order=order
        ),
        values=values,
    )


def test_catalog_addition_declaration_and_example_execute() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "class_function.add.compute"
    )
    request = tool.request_type.model_validate(
        {
            "phi": TRIVIAL.model_dump(),
            "psi": STANDARD.model_dump(),
        }
    )
    result = tool.run(request)
    assert result == class_function_add(TRIVIAL, STANDARD)
    assert tuple(value.coefficients[0].as_fraction() for value in result.values) == (
        Fraction(3),
        Fraction(1),
        Fraction(0),
    )


def test_catalog_declaration_and_example_execute() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "class_function.pointwise_multiply.compute"
    )
    request = tool.request_type.model_validate_json(
        encode_strict_json(
            {
                "phi": TRIVIAL.model_dump(mode="json"),
                "psi": STANDARD.model_dump(mode="json"),
            }
        ),
        strict=True,
    )
    result = tool.run(request)
    assert isinstance(result, FiniteClassFunction)
    assert result.values == STANDARD.values


TRIVIAL = _function(SIZES, (_value(1, (1,)),) * 3)
SIGN = _function(SIZES, (_value(1, (1,)), _value(1, (-1,)), _value(1, (1,))))
STANDARD = _function(SIZES, (_value(1, (2,)), _value(1, (0,)), _value(1, (-1,))))
