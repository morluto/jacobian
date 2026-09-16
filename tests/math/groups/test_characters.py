"""Tests for class_function.inner_product.compute."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import parse_operation_input
from jacobian.math.groups.characters._cyclotomic import (
    add_values,
    conjugate_value,
    value_from_power,
)
from jacobian.math.groups.characters._models import (
    MAX_CLASS_COUNT,
    ClassAxis,
    ClassFunctionInnerProductRequest,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters._tools import TOOLS
from jacobian.math.groups.characters.operations import class_function_inner_product

OPERATION_ID = "class_function.inner_product.compute"


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


# S3 conjugacy classes: identity (size 1), transpositions (size 3), 3-cycles (2).
S3_SIZES = (1, 3, 2)
S3_TRIVIAL = _class_function(
    1, S3_SIZES, (_value(1, (1,)), _value(1, (1,)), _value(1, (1,)))
)
S3_SIGN = _class_function(
    1, S3_SIZES, (_value(1, (1,)), _value(1, (-1,)), _value(1, (1,)))
)
S3_STANDARD = _class_function(
    1, S3_SIZES, (_value(1, (2,)), _value(1, (0,)), _value(1, (-1,)))
)


def _inner(phi: FiniteClassFunction, psi: FiniteClassFunction) -> CyclotomicValue:
    return class_function_inner_product(
        ClassFunctionInnerProductRequest(phi=phi, psi=psi)
    ).inner_product


def _is_rational(value: CyclotomicValue, rational: int | Fraction) -> bool:
    return value.order == 1 and value.coefficients[0].as_fraction() == Fraction(
        rational
    )


class TestKnownAnswers:
    def test_trivial_and_sign_are_orthogonal(self) -> None:
        assert _is_rational(_inner(S3_TRIVIAL, S3_SIGN), 0)

    def test_trivial_is_unit_norm(self) -> None:
        assert _is_rational(_inner(S3_TRIVIAL, S3_TRIVIAL), 1)

    def test_sign_is_unit_norm(self) -> None:
        assert _is_rational(_inner(S3_SIGN, S3_SIGN), 1)

    def test_standard_character_is_unit_norm(self) -> None:
        assert _is_rational(_inner(S3_STANDARD, S3_STANDARD), 1)

    def test_distinct_irreducibles_are_orthogonal(self) -> None:
        assert _is_rational(_inner(S3_TRIVIAL, S3_STANDARD), 0)
        assert _is_rational(_inner(S3_SIGN, S3_STANDARD), 0)

    def test_contribution_table_is_complete(self) -> None:
        result = class_function_inner_product(
            ClassFunctionInnerProductRequest(phi=S3_TRIVIAL, psi=S3_SIGN)
        )
        assert tuple(row.class_index for row in result.contributions) == (0, 1, 2)
        assert tuple(row.class_size for row in result.contributions) == S3_SIZES
        terms = [
            row.weighted_product.coefficients[0].as_fraction()
            for row in result.contributions
        ]
        assert terms == [Fraction(1), Fraction(-3), Fraction(2)]


class TestRootOfUnityValues:
    # The four linear characters of the cyclic group C4, class sizes all 1.
    C4_SIZES = (1, 1, 1, 1)

    @staticmethod
    def _character(exponent: int) -> FiniteClassFunction:
        values = tuple(
            _from_reduced(4, value_from_power(4, exponent * index))
            for index in range(4)
        )
        return _class_function(4, TestRootOfUnityValues.C4_SIZES, values)

    def test_linear_characters_are_orthonormal(self) -> None:
        for left in range(4):
            for right in range(4):
                result = _inner(self._character(left), self._character(right))
                assert result.order == 4
                expected = 1 if left == right else 0
                assert result.coefficients[0].as_fraction() == Fraction(expected)
                assert all(
                    coefficient.as_fraction() == 0
                    for coefficient in result.coefficients[1:]
                )


class TestDefiningInvariants:
    def test_conjugate_symmetry(self) -> None:
        forward = _inner(S3_TRIVIAL, S3_STANDARD)
        backward = _inner(S3_STANDARD, S3_TRIVIAL)
        order = forward.order
        conjugated = conjugate_value(
            order, tuple(c.as_fraction() for c in backward.coefficients)
        )
        assert tuple(c.as_fraction() for c in forward.coefficients) == conjugated

    def test_hermitian_conjugate_symmetry_with_root_of_unity(self) -> None:
        chi_one = TestRootOfUnityValues._character(1)
        chi_three = TestRootOfUnityValues._character(3)
        forward = _inner(chi_one, chi_three)
        backward = _inner(chi_three, chi_one)
        conjugated = conjugate_value(
            4, tuple(c.as_fraction() for c in backward.coefficients)
        )
        assert tuple(c.as_fraction() for c in forward.coefficients) == conjugated

    def test_left_linearity_in_the_second_argument(self) -> None:
        doubled = _class_function(
            1,
            S3_SIZES,
            (_value(1, (2,)), _value(1, (2,)), _value(1, (2,))),
        )
        baseline = _inner(S3_SIGN, S3_TRIVIAL)
        scaled = _inner(S3_SIGN, doubled)
        assert (
            scaled.coefficients[0].as_fraction()
            == 2 * baseline.coefficients[0].as_fraction()
        )

    def test_regular_character_norm_is_the_group_order(self) -> None:
        regular = _class_function(
            1, S3_SIZES, (_value(1, (6,)), _value(1, (0,)), _value(1, (0,)))
        )
        assert _is_rational(_inner(regular, regular), 6)

    def test_additive_value_arithmetic_matches_manual_sum(self) -> None:
        left = _value(1, (1,))
        right = _value(1, (2,))
        combined = add_values(1, (Fraction(1),), (Fraction(2),))
        assert combined == (Fraction(3),)
        assert _is_rational(_class_function(1, (1,), (left,)).values[0], 1)
        assert _is_rational(_class_function(1, (1,), (right,)).values[0], 2)


class TestBoundariesAndAdversarial:
    def test_trivial_group_single_class(self) -> None:
        phi = _class_function(1, (1,), (_value(1, (3,)),))
        psi = _class_function(1, (1,), (_value(1, (3,)),))
        assert _is_rational(_inner(phi, psi), 9)

    def test_mismatched_axis_is_rejected(self) -> None:
        other = _class_function(1, (2, 4), (_value(1, (1,)), _value(1, (1,))))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            class_function_inner_product(
                ClassFunctionInnerProductRequest(phi=S3_TRIVIAL, psi=other)
            )
        assert (
            exc_info.value.errors()[0]["type"]
            == "groups.characters.class_axis_mismatch"
        )

    def test_value_count_must_match_class_count(self) -> None:
        axis = ClassAxis(class_sizes=(1, 3, 2), group_order=6, cyclotomic_order=1)
        with pytest.raises(ValidationError):
            FiniteClassFunction(axis=axis, values=(_value(1, (1,)),))

    def test_value_cyclotomic_order_must_match_axis(self) -> None:
        axis = ClassAxis(class_sizes=(1, 1), group_order=2, cyclotomic_order=2)
        with pytest.raises(ValidationError):
            FiniteClassFunction(axis=axis, values=(_value(1, (1,)), _value(1, (1,))))

    def test_axis_group_order_must_equal_class_size_sum(self) -> None:
        with pytest.raises(ValidationError):
            ClassAxis(class_sizes=(1, 2), group_order=4, cyclotomic_order=1)


class TestEnvelope:
    def test_class_count_above_envelope_is_a_resource_rejection(self) -> None:
        sizes = (1,) * (MAX_CLASS_COUNT + 1)
        axis = ClassAxis.model_construct(
            class_sizes=sizes,
            group_order=sum(sizes),
            cyclotomic_order=1,
        )
        values = (_value(1, (1,)),) * len(sizes)
        phi = FiniteClassFunction.model_construct(axis=axis, values=values)
        request = ClassFunctionInnerProductRequest.model_construct(phi=phi, psi=phi)
        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            class_function_inner_product(request)
        assert (
            exc_info.value.errors()[0]["type"]
            == "groups.characters.class_count_exceeds_envelope"
        )


class TestParityAndSerialization:
    def test_native_matches_catalog_tool(self) -> None:
        tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
        request = ClassFunctionInnerProductRequest(phi=S3_TRIVIAL, psi=S3_STANDARD)
        native = class_function_inner_product(request)
        catalog = tool.run(request)
        assert native.model_dump() == catalog.model_dump()

    def test_request_round_trip_through_json(self) -> None:
        request = ClassFunctionInnerProductRequest(phi=S3_TRIVIAL, psi=S3_SIGN)
        encoded = request.model_dump_json()
        assert ClassFunctionInnerProductRequest.model_validate_json(encoded) == request

    def test_result_round_trip_through_json(self) -> None:
        result = class_function_inner_product(
            ClassFunctionInnerProductRequest(phi=S3_TRIVIAL, psi=S3_SIGN)
        )
        encoded = result.model_dump_json()
        parsed = type(result).model_validate_json(encoded)
        assert parsed.model_dump() == result.model_dump()

    def test_catalog_example_is_executable(self) -> None:
        tool = next(tool for tool in TOOLS if tool.operation_id == OPERATION_ID)
        request = parse_operation_input(tool.request_type, tool.examples[0].input)
        result = tool.run(request)
        assert result.inner_product.order == 1
        assert result.inner_product.coefficients[0].as_fraction() == 0

    def test_catalog_discovery(self) -> None:
        ids = {tool.operation_id for tool in BUILTIN_TOOLS}
        assert OPERATION_ID in ids
        assert "number_theory.character.inner_product.compute" not in ids
