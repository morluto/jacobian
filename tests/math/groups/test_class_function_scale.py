"""Exact scalar action on finite-group class functions."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.characters._cyclotomic import value_from_power
from jacobian.math.groups.characters._models import (
    ClassAxis,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import class_function_scale


def _value(order: int, coefficients: tuple[int, ...]) -> CyclotomicValue:
    return CyclotomicValue(
        order=order,
        coefficients=tuple(
            CanonicalRational.from_fraction(Fraction(value)) for value in coefficients
        ),
    )


def test_cyclotomic_scalar_rotates_c3_linear_character_values() -> None:
    # Multiplication by zeta_3 takes (1, zeta_3, zeta_3^2) to
    # (zeta_3, zeta_3^2, 1), an independent value-from-power oracle.
    character = FiniteClassFunction(
        axis=ClassAxis(class_sizes=(1, 1, 1), group_order=3, cyclotomic_order=3),
        values=tuple(
            CyclotomicValue(
                order=3,
                coefficients=tuple(
                    CanonicalRational.from_fraction(value)
                    for value in value_from_power(3, exponent)
                ),
            )
            for exponent in range(3)
        ),
    )
    result = class_function_scale(character, _value(3, (0, 1)))
    assert result.axis == character.axis
    expected = tuple(value_from_power(3, exponent) for exponent in (1, 2, 3))
    assert (
        tuple(
            tuple(coefficient.as_fraction() for coefficient in value.coefficients)
            for value in result.values
        )
        == expected
    )


def test_s3_rational_scalar_and_json_roundtrip() -> None:
    function = FiniteClassFunction(
        axis=ClassAxis(class_sizes=(1, 3, 2), group_order=6, cyclotomic_order=1),
        values=(_value(1, (2,)), _value(1, (0,)), _value(1, (-1,))),
    )
    result = class_function_scale(function, _value(1, (-2,)))
    restored = FiniteClassFunction.model_validate_json(
        encode_strict_json(result.model_dump(mode="json"))
    )
    assert restored.axis == function.axis
    assert tuple(value.coefficients[0].as_fraction() for value in restored.values) == (
        Fraction(-4),
        Fraction(0),
        Fraction(2),
    )


def test_scalar_from_another_cyclotomic_field_is_rejected() -> None:
    function = FiniteClassFunction(
        axis=ClassAxis(class_sizes=(1,), group_order=1, cyclotomic_order=1),
        values=(_value(1, (1,)),),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        class_function_scale(function, _value(2, (1,)))
    assert error.value.errors()[0]["type"] == "groups.characters.scalar_field_mismatch"


def test_coefficient_growth_is_rejected_before_multiplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.groups.characters import operations

    huge = _value(1, (10**512 - 1,))
    function = FiniteClassFunction(
        axis=ClassAxis(class_sizes=(1,), group_order=1, cyclotomic_order=1),
        values=(huge,),
    )

    def forbidden(*args: object, **kwargs: object) -> tuple[Fraction, ...]:
        raise AssertionError("cyclotomic multiplication ran before admission")

    monkeypatch.setattr(operations, "multiply_values", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        class_function_scale(function, _value(1, (2,)))


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
