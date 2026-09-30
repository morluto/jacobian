"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/groups/test_class_function_conjugate.py``. The preamble below is carried over verbatim so the moved
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


def _value(order: int, coefficients: tuple[int, ...]) -> CyclotomicValue:
    return CyclotomicValue(
        order=order,
        coefficients=tuple(
            CanonicalRational.from_fraction(Fraction(coefficient))
            for coefficient in coefficients
        ),
    )


def _c3_character() -> FiniteClassFunction:
    # On C3, the first linear character is (1, zeta_3, zeta_3^2), with
    # zeta_3^2 = -1-zeta_3 in the distinguished power basis.
    return FiniteClassFunction(
        axis=ClassAxis(class_sizes=(1, 1, 1), group_order=3, cyclotomic_order=3),
        values=(_value(3, (1, 0)), _value(3, (0, 1)), _value(3, (-1, -1))),
    )


def _coefficients(function: FiniteClassFunction) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(coefficient.as_fraction() for coefficient in value.coefficients)
        for value in function.values
    )


def test_catalog_conjugation_example_survives_json_and_runs() -> None:
    tool = next(
        candidate
        for candidate in BUILTIN_TOOLS
        if candidate.operation_id == "class_function.conjugate.compute"
    )
    encoded = encode_strict_json(tool.examples[0].input)
    request = tool.request_type.model_validate_json(encoded, strict=True)
    result = tool.run(request)
    assert isinstance(result, FiniteClassFunction)
    assert result.axis.cyclotomic_order == 3
    assert _coefficients(result)[1] == (Fraction(-1), Fraction(-1))
