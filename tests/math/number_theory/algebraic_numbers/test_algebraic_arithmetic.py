"""Contract and mathematical tests for algebraic number arithmetic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.algebraic_numbers import (
    add_quadratic,
    multiply_quadratic,
)
from jacobian.math.number_theory.algebraic_numbers._models import (
    _MAX_RESULT_DIGITS,
    AlgebraicAdditionRequest,
    AlgebraicArithmeticRequest,
    AlgebraicMultiplicationRequest,
)
from jacobian.math.number_theory.algebraic_numbers._tools import (
    TOOLS,
    compute_algebraic_add,
    compute_algebraic_multiply,
)
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue


def _cr(value: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, den)


def _element(a: int, b: int, d: int) -> RealQuadraticValue:
    return RealQuadraticValue(
        rational_part=_cr(a),
        radical_coefficient=_cr(b),
        radicand=d,
    )


def _req(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    request_type: type[AlgebraicArithmeticRequest],
) -> AlgebraicArithmeticRequest:
    return request_type(
        left=_element(*left),
        right=_element(*right),
    )


def _add_req(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> AlgebraicAdditionRequest:
    return AlgebraicAdditionRequest(left=_element(*left), right=_element(*right))


def _mul_req(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> AlgebraicMultiplicationRequest:
    return AlgebraicMultiplicationRequest(left=_element(*left), right=_element(*right))


def test_addition_is_component_wise() -> None:
    result = compute_algebraic_add(_add_req((1, 1, 2), (3, 2, 2)))
    assert result.rational_part.as_fraction() == 4
    assert result.radical_coefficient.as_fraction() == 3
    assert result.radicand == 2
    assert add_quadratic(_element(1, 1, 2), _element(3, 2, 2)) == result


def test_addition_commutativity() -> None:
    left = compute_algebraic_add(_add_req((1, 3, 5), (2, 1, 5)))
    right = compute_algebraic_add(_add_req((2, 1, 5), (1, 3, 5)))
    assert left == right


def test_addition_identity() -> None:
    result = compute_algebraic_add(_add_req((7, 3, 2), (0, 0, 2)))
    assert result.rational_part.as_fraction() == 7
    assert result.radical_coefficient.as_fraction() == 3


def test_multiplication_distributes_over_addition() -> None:
    # (a+b)(c+d) = ac + (ad+bc) + bd for sqrt(d) field
    # (1 + sqrt(2)) * (1 - sqrt(2)) = 1 - 2 = -1
    result = compute_algebraic_multiply(_mul_req((1, 1, 2), (1, -1, 2)))
    assert result.rational_part.as_fraction() == -1
    assert result.radical_coefficient.as_fraction() == 0
    assert multiply_quadratic(_element(1, 1, 2), _element(1, -1, 2)) == result


def test_multiplication_commutativity() -> None:
    left = compute_algebraic_multiply(_mul_req((3, 1, 2), (1, 2, 2)))
    right = compute_algebraic_multiply(_mul_req((1, 2, 2), (3, 1, 2)))
    assert left == right


def test_multiplication_by_rational() -> None:
    # 2 * (1 + sqrt(2)) = 2 + 2*sqrt(2)
    # This is the exact error case from issue #916 where the model computed (2, 1) instead of (2, 2)
    result = compute_algebraic_multiply(_mul_req((2, 0, 2), (1, 1, 2)))
    assert result.rational_part.as_fraction() == 2
    assert result.radical_coefficient.as_fraction() == 2


def test_multiplication_by_rational_irrational() -> None:
    # (2, 0) * (2, 2) = (2*2 + 0*2*2, 2*2 + 0*2) = (4, 4)
    # This is the second error case from issue #916
    result = compute_algebraic_multiply(_mul_req((2, 0, 2), (2, 2, 2)))
    assert result.rational_part.as_fraction() == 4
    assert result.radical_coefficient.as_fraction() == 4


def test_fractional_coefficients() -> None:
    # (1/2 + sqrt(3)) * (1/2 - sqrt(3)) = 1/4 - 3 = -11/4
    left = RealQuadraticValue(
        rational_part=CanonicalRational.from_integer_ratio(1, 2),
        radical_coefficient=_cr(1),
        radicand=3,
    )
    right = RealQuadraticValue(
        rational_part=CanonicalRational.from_integer_ratio(1, 2),
        radical_coefficient=_cr(-1),
        radicand=3,
    )
    result = compute_algebraic_multiply(
        AlgebraicMultiplicationRequest(left=left, right=right)
    )
    assert result.rational_part.as_fraction() == pytest.approx(-11 / 4)
    # exact check via Fraction
    from fractions import Fraction

    assert result.rational_part.as_fraction() == Fraction(-11, 4)
    assert result.radical_coefficient.as_fraction() == 0


def test_mismatched_radicands_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AlgebraicArithmeticRequest(
            left=_element(1, 1, 2),
            right=_element(1, 1, 3),
        )
        # The shared base request still enforces one shared radicand.
    assert exc_info.value.errors()[0]["type"] == (
        "algebraic_number_arithmetic.radicands_must_match"
    )


def test_invalid_radicand_rejected() -> None:
    with pytest.raises((ValueError, ValidationError), match="radicand"):
        RealQuadraticValue(
            rational_part=_cr(1),
            radical_coefficient=_cr(1),
            radicand=1,
        )


def test_non_squarefree_radicand_rejected() -> None:
    with pytest.raises((ValueError, ValidationError), match="square-free"):
        RealQuadraticValue(
            rational_part=_cr(1),
            radical_coefficient=_cr(1),
            radicand=12,
        )
    with pytest.raises((ValueError, ValidationError), match="square-free"):
        RealQuadraticValue(
            rational_part=_cr(1),
            radical_coefficient=_cr(1),
            radicand=4,
        )


def test_result_remains_consumable() -> None:
    # Result of an operation must itself be a valid operand for a subsequent operation.
    first = compute_algebraic_add(_add_req((1, 1, 2), (3, 2, 2)))
    # Re-use the result as an operand in a second operation
    second_req = AlgebraicAdditionRequest(
        left=first,
        right=_element(0, 0, 2),
    )
    second = compute_algebraic_add(second_req)
    assert second.rational_part.as_fraction() == 4
    assert second.radical_coefficient.as_fraction() == 3


def test_operations_in_catalog() -> None:
    ids = {tool.operation_id for tool in TOOLS}
    assert "algebraic_number.add.compute" in ids
    assert "algebraic_number.multiply.compute" in ids


def test_addition_admits_representable_sums_with_overflowing_products() -> None:
    # Adding two 10**200 rational parts yields a 201-digit sum, well within
    # the 256-digit result bound; the (unused) product must not matter.
    big = 10**200
    request = _add_req((big, 0, 2), (big, 0, 2))
    result = compute_algebraic_add(request)
    assert result.rational_part.as_fraction() == 2 * big


def test_multiplication_admits_representable_products_with_overflowing_sums() -> None:
    # Multiplying (10**256 - 1) by 1 yields a representable 256-digit
    # product; the (unused) component-wise sum must not matter.
    huge = 10**256 - 1
    request = _mul_req((huge, 0, 2), (1, 0, 2))
    result = compute_algebraic_multiply(request)
    assert result.rational_part.as_fraction() == huge


def test_multiplication_still_rejects_unrepresentable_products() -> None:
    # (10**200)^2 exceeds the 256-digit result bound for multiply...
    big = 10**200
    request = AlgebraicMultiplicationRequest(
        left=_element(big, 0, 2),
        right=_element(big, 0, 2),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_algebraic_multiply(request)
    assert exc_info.value.errors()[0]["type"] == (
        "algebraic_number_arithmetic.multiplication_result_exceeds_bound"
    )
    # ...while the same operands remain valid input for addition.
    assert compute_algebraic_add(_add_req((big, 0, 2), (big, 0, 2)))


def test_addition_still_rejects_unrepresentable_sums() -> None:
    # Two maximal 256-digit rational parts sum to a 257-digit value
    # beyond the result bound.
    huge = 10**256 - 1
    request = AlgebraicAdditionRequest(
        left=_element(huge, 0, 2),
        right=_element(huge, 0, 2),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_algebraic_add(request)
    assert exc_info.value.errors()[0]["type"] == (
        "algebraic_number_arithmetic.addition_result_exceeds_bound"
    )


def test_operation_declarations_expose_operand_preconditions() -> None:
    """Both declarations state the shared-radicand and result-growth rules."""

    tools = {tool.operation_id: tool for tool in TOOLS}
    for operation_id in (
        "algebraic_number.add.compute",
        "algebraic_number.multiply.compute",
    ):
        tool = tools[operation_id]
        description = tool.description.lower()
        assert "same square-free radicand" in description
        assert f"{_MAX_RESULT_DIGITS}-digit" in description
        schema = tool.request_type.model_json_schema()
        properties = schema["properties"]
        for side in ("left", "right"):
            field_description = properties[side]["description"].lower()
            assert "radicand" in field_description
            assert "square-free" in field_description
        for example_spec in tool.examples:
            text = str(example_spec.description).lower()
            assert "square-free" in text
