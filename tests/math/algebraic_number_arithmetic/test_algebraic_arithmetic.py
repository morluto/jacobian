"""Contract and mathematical tests for algebraic number arithmetic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.algebraic_number_arithmetic._models import (
    AlgebraicArithmeticRequest,
    QuadraticElement,
)
from jacobian.math.algebraic_number_arithmetic._operations import (
    compute_algebraic_add,
    compute_algebraic_multiply,
)
from jacobian.math.algebraic_number_arithmetic._tools import TOOLS


def _cr(value: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, den)


def _element(a: int, b: int, d: int) -> QuadraticElement:
    return QuadraticElement(
        rational_part=_cr(a),
        radical_coefficient=_cr(b),
        radicand=d,
    )


def _req(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> AlgebraicArithmeticRequest:
    return AlgebraicArithmeticRequest(
        left=_element(*left),
        right=_element(*right),
    )


def test_addition_is_component_wise() -> None:
    result = compute_algebraic_add(_req((1, 1, 2), (3, 2, 2)))
    assert result.rational_part.as_fraction() == 4
    assert result.radical_coefficient.as_fraction() == 3
    assert result.radicand == 2


def test_addition_commutativity() -> None:
    left = compute_algebraic_add(_req((1, 3, 5), (2, 1, 5)))
    right = compute_algebraic_add(_req((2, 1, 5), (1, 3, 5)))
    assert left == right


def test_addition_identity() -> None:
    result = compute_algebraic_add(_req((7, 3, 2), (0, 0, 2)))
    assert result.rational_part.as_fraction() == 7
    assert result.radical_coefficient.as_fraction() == 3


def test_multiplication_distributes_over_addition() -> None:
    # (a+b)(c+d) = ac + (ad+bc) + bd for sqrt(d) field
    # (1 + sqrt(2)) * (1 - sqrt(2)) = 1 - 2 = -1
    result = compute_algebraic_multiply(_req((1, 1, 2), (1, -1, 2)))
    assert result.rational_part.as_fraction() == -1
    assert result.radical_coefficient.as_fraction() == 0


def test_multiplication_commutativity() -> None:
    left = compute_algebraic_multiply(_req((3, 1, 2), (1, 2, 2)))
    right = compute_algebraic_multiply(_req((1, 2, 2), (3, 1, 2)))
    assert left == right


def test_multiplication_by_rational() -> None:
    # 2 * (1 + sqrt(2)) = 2 + 2*sqrt(2)
    # This is the exact error case from issue #916 where the model computed (2, 1) instead of (2, 2)
    result = compute_algebraic_multiply(_req((2, 0, 2), (1, 1, 2)))
    assert result.rational_part.as_fraction() == 2
    assert result.radical_coefficient.as_fraction() == 2


def test_multiplication_by_rational_irrational() -> None:
    # (2, 0) * (2, 2) = (2*2 + 0*2*2, 2*2 + 0*2) = (4, 4)
    # This is the second error case from issue #916
    result = compute_algebraic_multiply(_req((2, 0, 2), (2, 2, 2)))
    assert result.rational_part.as_fraction() == 4
    assert result.radical_coefficient.as_fraction() == 4


def test_fractional_coefficients() -> None:
    # (1/2 + sqrt(3)) * (1/2 - sqrt(3)) = 1/4 - 3 = -11/4
    left = QuadraticElement(
        rational_part=CanonicalRational.from_integer_ratio(1, 2),
        radical_coefficient=_cr(1),
        radicand=3,
    )
    right = QuadraticElement(
        rational_part=CanonicalRational.from_integer_ratio(1, 2),
        radical_coefficient=_cr(-1),
        radicand=3,
    )
    result = compute_algebraic_multiply(
        AlgebraicArithmeticRequest(left=left, right=right)
    )
    assert result.rational_part.as_fraction() == pytest.approx(-11 / 4)  # type: ignore[comparison-overlap]
    # exact check via Fraction
    from fractions import Fraction

    assert result.rational_part.as_fraction() == Fraction(-11, 4)
    assert result.radical_coefficient.as_fraction() == 0


def test_mismatched_radicands_rejected() -> None:
    with pytest.raises((ValueError, ValidationError), match="same quadratic field"):
        AlgebraicArithmeticRequest(
            left=_element(1, 1, 2),
            right=_element(1, 1, 3),
        )


def test_invalid_radicand_rejected() -> None:
    with pytest.raises((ValueError, ValidationError), match="radicand"):
        QuadraticElement(
            rational_part=_cr(1),
            radical_coefficient=_cr(1),
            radicand=1,
        )


def test_non_squarefree_radicand_rejected() -> None:
    with pytest.raises((ValueError, ValidationError), match="square-free"):
        QuadraticElement(
            rational_part=_cr(1),
            radical_coefficient=_cr(1),
            radicand=12,
        )
    with pytest.raises((ValueError, ValidationError), match="square-free"):
        QuadraticElement(
            rational_part=_cr(1),
            radical_coefficient=_cr(1),
            radicand=4,
        )


def test_result_remains_consumable() -> None:
    # Result of an operation must itself be a valid operand for a subsequent operation.
    first = compute_algebraic_add(_req((1, 1, 2), (3, 2, 2)))
    # Re-use the result as an operand in a second operation
    second_req = AlgebraicArithmeticRequest(
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
