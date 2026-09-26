"""Contracts and exact finite-field oracle for function-field addition."""

from __future__ import annotations

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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


def test_rational_function_sum_is_reduced_and_retains_its_field_parent() -> None:
    field = _rational_field(7)
    left = _element(field, (_rational(7, (2,), (2, 2)),))
    right = _element(field, (_rational(7, (1,), (2, 1)),))

    result = function_field_element_add(left, right)

    assert result.field == field
    assert result.coordinates == (_rational(7, (3, 2), (2, 3, 1)),)


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


def test_addition_matches_independent_finite_field_specializations() -> None:
    prime = 7
    field = _quadratic_field(prime)
    x_plus_one_over_x_plus_two = _rational(prime, (1, 1), (2, 1))
    inverse_x_plus_one = _rational(prime, (1,), (1, 1))
    two_over_x_plus_two = _rational(prime, (2,), (2, 1))
    x_over_x_plus_one = _rational(prime, (0, 1), (1, 1))
    left = _element(field, (x_plus_one_over_x_plus_two, inverse_x_plus_one))
    right = _element(field, (two_over_x_plus_two, x_over_x_plus_one))

    result = function_field_element_add(left, right)

    # Evaluate in GF(7) at every available rational point in this small sample
    # whose denominators are nonzero. This oracle uses field specialization and
    # y^2=x directly, independently of the rational-function addition kernel.
    for point in (0, 1, 2, 4):
        roots = tuple(value for value in range(prime) if value * value % prime == point)
        assert roots
        for generator_value in roots:
            expected = (
                _evaluate_quadratic_element(left, point, generator_value)
                + _evaluate_quadratic_element(right, point, generator_value)
            ) % prime
            assert (
                _evaluate_quadratic_element(result, point, generator_value) == expected
            )


def test_addition_rejects_different_field_parents() -> None:
    left = _element(_rational_field(5), (_rational(5, (1,)),))
    right = _element(_rational_field(7), (_rational(7, (1,)),))

    with pytest.raises(OperationDomainValidationError) as error:
        function_field_element_add(left, right)

    assert error.value.errors()[0]["type"] == "function_field.element_field_mismatch"


def test_growth_admission_precedes_coordinate_addition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.function_fields.operations as operations

    field = _rational_field(7)
    degree_twelve = (0,) * 12 + (1,)
    left_denominator = (1,) + (0,) * 11 + (1,)
    right_denominator = (1, 1) + (0,) * 10 + (1,)
    left = _element(field, (_rational(7, degree_twelve, left_denominator),))
    right = _element(field, (_rational(7, (1,), right_denominator),))

    def unexpected_addition(*_args, **_kwargs):
        raise AssertionError("admission must run before rational-function addition")

    monkeypatch.setattr(operations, "rf_add", unexpected_addition)
    with pytest.raises(OperationResourceAdmissionError) as error:
        function_field_element_add(left, right)

    assert (
        error.value.errors()[0]["type"]
        == "function_field.addition_coefficient_growth_exceeds_envelope"
    )


def test_work_and_output_admission_precede_coordinate_addition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.function_fields.operations as operations

    field = _rational_field(7)
    left = _element(field, (_rational(7, (1, 1)),))
    right = _element(field, (_rational(7, (2, 1)),))

    def unexpected_addition(*_args, **_kwargs):
        raise AssertionError("admission must run before rational-function addition")

    monkeypatch.setattr(operations, "rf_add", unexpected_addition)
    monkeypatch.setattr(operations, "MAX_ELEMENT_ADDITION_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError) as work_error:
        function_field_element_add(left, right)
    assert (
        work_error.value.errors()[0]["type"]
        == "function_field.element_addition_work_exceeds_envelope"
    )

    monkeypatch.setattr(operations, "MAX_ELEMENT_ADDITION_WORK", 2_000_000)
    monkeypatch.setattr(operations, "MAX_ELEMENT_VALUE_BYTES", 1)
    with pytest.raises(OperationResourceAdmissionError) as output_error:
        function_field_element_add(left, right)
    assert (
        output_error.value.errors()[0]["type"]
        == "function_field.element_addition_output_exceeds_envelope"
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
