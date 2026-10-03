"""Catalog discovery check for the group character operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.groups.characters._models import (
    ClassAxis,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import class_function_inner_product

OPERATION_ID = "class_function.inner_product.compute"
S3_SIZES = (1, 3, 2)


def _value(order: int, coefficients: tuple[Fraction | int, ...]) -> CyclotomicValue:
    return CyclotomicValue(
        order=order,
        coefficients=tuple(
            CanonicalRational.from_fraction(Fraction(coefficient))
            for coefficient in coefficients
        ),
    )


def _from_reduced(order: int, coefficients: tuple[Fraction, ...]) -> CyclotomicValue:
    return CyclotomicValue(
        order=order,
        coefficients=tuple(
            CanonicalRational.from_fraction(coefficient) for coefficient in coefficients
        ),
    )


def _class_function(
    order: int, class_sizes: tuple[int, ...], values: tuple[CyclotomicValue, ...]
) -> FiniteClassFunction:
    axis = ClassAxis(
        class_sizes=class_sizes,
        group_order=sum(class_sizes),
        cyclotomic_order=order,
    )
    return FiniteClassFunction(axis=axis, values=values)


def _inner(phi: FiniteClassFunction, psi: FiniteClassFunction) -> CyclotomicValue:
    return class_function_inner_product(phi, psi).inner_product


def _is_rational(value: CyclotomicValue, rational: int | Fraction) -> bool:
    return value.order == 1 and value.coefficients[0].as_fraction() == Fraction(
        rational
    )


def test_catalog_discovery() -> None:
    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in ids
    assert "number_theory.character.inner_product.compute" not in ids


S3_TRIVIAL = _class_function(
    1, S3_SIZES, (_value(1, (1,)), _value(1, (1,)), _value(1, (1,)))
)
S3_SIGN = _class_function(
    1, S3_SIZES, (_value(1, (1,)), _value(1, (-1,)), _value(1, (1,)))
)
S3_STANDARD = _class_function(
    1, S3_SIZES, (_value(1, (2,)), _value(1, (0,)), _value(1, (-1,)))
)
