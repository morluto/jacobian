"""Exact pointwise class-function multiplication and tensor characters."""

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
from jacobian.math.groups.characters.operations import (
    class_function_add,
    class_function_inner_product,
    class_function_pointwise_product,
)


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


SIZES = (1, 3, 2)
TRIVIAL = _function(SIZES, (_value(1, (1,)),) * 3)
SIGN = _function(SIZES, (_value(1, (1,)), _value(1, (-1,)), _value(1, (1,))))
STANDARD = _function(SIZES, (_value(1, (2,)), _value(1, (0,)), _value(1, (-1,))))


def test_standard_tensor_square_has_s3_character_decomposition() -> None:
    # The S3 tensor-square identity is chi_std^2 = chi_triv + chi_sign + chi_std.
    # Its values are (4, 0, 1); the RHS independently gives the same vector.
    square = class_function_pointwise_product(STANDARD, STANDARD)
    assert tuple(value.coefficients[0].as_fraction() for value in square.values) == (
        Fraction(4),
        Fraction(0),
        Fraction(1),
    )
    rhs = tuple(
        sum(
            character.values[index].coefficients[0].as_fraction()
            for character in (TRIVIAL, SIGN, STANDARD)
        )
        for index in range(3)
    )
    assert tuple(value.coefficients[0].as_fraction() for value in square.values) == rhs
    assert square.axis == STANDARD.axis


def test_trivial_plus_standard_is_s3_class_function_sum() -> None:
    result = class_function_add(TRIVIAL, STANDARD)
    assert tuple(value.coefficients[0].as_fraction() for value in result.values) == (
        Fraction(3),
        Fraction(1),
        Fraction(0),
    )
    assert result.axis == STANDARD.axis


def test_cyclotomic_linear_characters_multiply_to_trivial_after_roundtrip() -> None:
    # C4 has four singleton classes. chi_1(g^j)=zeta_4^j and
    # chi_3(g^j)=zeta_4^(3j), so their product is chi_0 at each element.
    values_one = tuple(
        CyclotomicValue(
            order=4,
            coefficients=tuple(
                CanonicalRational.from_fraction(coefficient)
                for coefficient in value_from_power(4, exponent)
            ),
        )
        for exponent in range(4)
    )
    values_three = tuple(
        CyclotomicValue(
            order=4,
            coefficients=tuple(
                CanonicalRational.from_fraction(coefficient)
                for coefficient in value_from_power(4, 3 * exponent)
            ),
        )
        for exponent in range(4)
    )
    axis = ClassAxis(class_sizes=(1, 1, 1, 1), group_order=4, cyclotomic_order=4)
    left = FiniteClassFunction(axis=axis, values=values_one)
    right = FiniteClassFunction(axis=axis, values=values_three)
    product = class_function_pointwise_product(left, right)
    transported = FiniteClassFunction.model_validate_json(
        encode_strict_json(product.model_dump(mode="json"))
    )
    assert transported.axis == axis
    assert tuple(
        value.coefficients[0].as_fraction() for value in transported.values
    ) == (Fraction(1), Fraction(1), Fraction(1), Fraction(1))
    assert all(
        all(coefficient.as_fraction() == 0 for coefficient in value.coefficients[1:])
        for value in transported.values
    )


def test_product_composes_with_inner_product_on_unchanged_axis() -> None:
    product = class_function_pointwise_product(STANDARD, STANDARD)
    result = class_function_inner_product(product, TRIVIAL)
    # (4,0,1) paired with the trivial character has weighted average one.
    assert result.inner_product.coefficients[0].as_fraction() == 1


def test_axis_mismatch_is_rejected() -> None:
    different_axis = _function((1, 1, 4), STANDARD.values)
    with pytest.raises(OperationDomainValidationError) as error:
        class_function_pointwise_product(STANDARD, different_axis)
    assert error.value.errors()[0]["type"] == "groups.characters.class_axis_mismatch"


def test_addition_rejects_unequal_canonical_axes() -> None:
    different_axis = _function((1, 1, 4), STANDARD.values)
    with pytest.raises(OperationDomainValidationError) as error:
        class_function_add(STANDARD, different_axis)
    assert error.value.errors()[0]["type"] == "groups.characters.class_axis_mismatch"


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


def test_addition_admits_coefficient_growth_before_exact_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.groups.characters import operations

    huge = _function((1,), (_value(1, (10**512 - 1,)),))

    def forbidden_add(*args: object, **kwargs: object) -> tuple[Fraction, ...]:
        raise AssertionError("exact addition ran before admission")

    monkeypatch.setattr(operations, "add_values", forbidden_add)
    with pytest.raises(OperationResourceAdmissionError) as error:
        class_function_add(huge, huge)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.add_output_digits_exceed_envelope"
    )


def test_exact_512_digit_result_boundary_is_admitted() -> None:
    value = 10**256 - 1
    function = _function((1,), (_value(1, (value,)),))
    result = class_function_pointwise_product(function, function)
    assert len(str(result.values[0].coefficients[0].num)) == 512


def test_predicted_coefficient_growth_is_rejected_before_multiplication() -> None:
    value = 10**257 - 1
    function = _function((1,), (_value(1, (value,)),))
    with pytest.raises(OperationResourceAdmissionError) as error:
        class_function_pointwise_product(function, function)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.pointwise_product_output_digits_exceed_envelope"
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


def _forged(
    values: tuple[CyclotomicValue, ...], sizes: tuple[int, ...] = SIZES, order: int = 1
) -> FiniteClassFunction:
    return FiniteClassFunction.model_construct(
        axis=ClassAxis(
            class_sizes=sizes, group_order=sum(sizes), cyclotomic_order=order
        ),
        values=values,
    )


@pytest.mark.parametrize(
    "operation", [class_function_add, class_function_pointwise_product]
)
def test_native_operations_reject_forged_empty_class_functions(
    operation,
) -> None:
    forged = _forged(())
    with pytest.raises(OperationDomainValidationError) as error:
        operation(forged, forged)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.invalid_class_function"
    )


@pytest.mark.parametrize(
    "operation",
    [
        class_function_add,
        class_function_pointwise_product,
        class_function_inner_product,
    ],
)
def test_native_operations_reject_values_from_another_axis_length(operation) -> None:
    # One value on a three-class axis: same-length zips would otherwise pass
    # and the kernel would forge another axis-bound invalid result.
    forged = _forged((_value(1, (1,)),))
    with pytest.raises(OperationDomainValidationError) as error:
        operation(forged, forged)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.invalid_class_function"
    )


def test_pointwise_product_rejects_values_outside_the_axis_field() -> None:
    forged = _forged((_value(2, (1,)),) * 3)
    with pytest.raises(OperationDomainValidationError) as error:
        class_function_pointwise_product(forged, forged)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.invalid_class_function"
    )


def test_addition_rejects_values_outside_the_axis_field() -> None:
    forged = _forged((_value(2, (1,)),) * 3)
    with pytest.raises(OperationDomainValidationError) as error:
        class_function_add(forged, forged)
    assert error.value.errors()[0]["type"] == (
        "groups.characters.invalid_class_function"
    )
