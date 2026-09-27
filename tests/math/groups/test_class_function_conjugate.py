"""Independent exact checks for class-function cyclotomic conjugation."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.characters._models import (
    ClassAxis,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import (
    class_function_conjugate,
    class_function_inner_product,
    class_function_pointwise_product,
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


def test_c3_character_conjugation_is_the_inverse_character_exactly() -> None:
    character = _c3_character()
    conjugate = class_function_conjugate(character)
    assert conjugate.axis == character.axis
    assert _coefficients(conjugate) == (
        (Fraction(1), Fraction(0)),
        (Fraction(-1), Fraction(-1)),
        (Fraction(0), Fraction(1)),
    )


def test_conjugation_is_an_involution_and_composes_with_the_class_algebra() -> None:
    character = _c3_character()
    inverse_character = class_function_conjugate(character)
    assert class_function_conjugate(inverse_character) == character

    product = class_function_pointwise_product(character, inverse_character)
    assert _coefficients(product) == (
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(0)),
    )
    pairing = class_function_inner_product(product, product)
    assert pairing.inner_product == _value(3, (1, 0))


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


def test_coefficient_height_is_admitted_before_cyclotomic_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = FiniteClassFunction(
        axis=ClassAxis(class_sizes=(1,), group_order=1, cyclotomic_order=1),
        values=(_value(1, (10**512,)),),
    )

    def fail_before_expansion(*args: object, **kwargs: object) -> tuple[Fraction, ...]:
        raise AssertionError("oversized input entered cyclotomic conjugation")

    monkeypatch.setattr(
        "jacobian.math.groups.characters.operations.conjugate_value",
        fail_before_expansion,
    )
    with pytest.raises(OperationResourceAdmissionError):
        class_function_conjugate(oversized)
