"""Contract and mathematical tests for algebraic number arithmetic."""

from __future__ import annotations

import pytest

from jacobian.math.algebraic_number_arithmetic._models import (
    AlgebraicArithmeticRequest,
    QuadraticElement,
)
from jacobian.math.algebraic_number_arithmetic._operations import (
    compute_algebraic_add,
    compute_algebraic_multiply,
)
from jacobian.math.algebraic_number_arithmetic._tools import TOOLS


def _element(a: int, b: int, d: int) -> QuadraticElement:
    return QuadraticElement(
        rational_part_num=a,
        irrational_part_num=b,
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
    assert result.rational_part_num == 4
    assert result.rational_part_den == 1
    assert result.irrational_part_num == 3
    assert result.irrational_part_den == 1
    assert result.radicand == 2


def test_addition_commutativity() -> None:
    left = compute_algebraic_add(_req((1, 3, 5), (2, 1, 5)))
    right = compute_algebraic_add(_req((2, 1, 5), (1, 3, 5)))
    assert left == right


def test_addition_identity() -> None:
    result = compute_algebraic_add(_req((7, 3, 2), (0, 0, 2)))
    assert result.rational_part_num == 7
    assert result.irrational_part_num == 3


def test_multiplication_distributes_over_addition() -> None:
    # (a+b)(c+d) = ac + (ad+bc) + bd for sqrt(d) field
    # (1 + sqrt(2)) * (1 - sqrt(2)) = 1 - 2 = -1
    result = compute_algebraic_multiply(_req((1, 1, 2), (1, -1, 2)))
    assert result.rational_part_num == -1
    assert result.rational_part_den == 1
    assert result.irrational_part_num == 0
    assert result.irrational_part_den == 1


def test_multiplication_commutativity() -> None:
    left = compute_algebraic_multiply(_req((3, 1, 2), (1, 2, 2)))
    right = compute_algebraic_multiply(_req((1, 2, 2), (3, 1, 2)))
    assert left == right


def test_multiplication_by_rational() -> None:
    # 2 * (1 + sqrt(2)) = 2 + 2*sqrt(2)
    # This is the exact error case from issue #916 where the model computed (2, 1) instead of (2, 2)
    result = compute_algebraic_multiply(_req((2, 0, 2), (1, 1, 2)))
    assert result.rational_part_num == 2
    assert result.irrational_part_num == 2
    assert result.irrational_part_den == 1


def test_multiplication_by_rational_irrational() -> None:
    # (2, 0) * (2, 2) = (2*2 + 0*2*2, 2*2 + 0*2) = (4, 4)
    # This is the second error case from issue #916
    result = compute_algebraic_multiply(_req((2, 0, 2), (2, 2, 2)))
    assert result.rational_part_num == 4
    assert result.irrational_part_num == 4


def test_fractional_coefficients() -> None:
    # (1/2 + sqrt(3)) * (1/2 - sqrt(3)) = 1/4 - 3 = -11/4
    left = QuadraticElement(
        rational_part_num=1,
        rational_part_den=2,
        irrational_part_num=1,
        radicand=3,
    )
    right = QuadraticElement(
        rational_part_num=1,
        rational_part_den=2,
        irrational_part_num=-1,
        radicand=3,
    )
    result = compute_algebraic_multiply(
        AlgebraicArithmeticRequest(left=left, right=right)
    )
    assert result.rational_part_num == -11
    assert result.rational_part_den == 4
    assert result.irrational_part_num == 0


def test_mismatched_radicands_rejected() -> None:
    with pytest.raises(ValueError, match="same quadratic field"):
        AlgebraicArithmeticRequest(
            left=_element(1, 1, 2),
            right=_element(1, 1, 3),
        )


def test_invalid_radicand_rejected() -> None:
    with pytest.raises(ValueError, match="radicand"):
        QuadraticElement(rational_part_num=1, irrational_part_num=1, radicand=1)


def test_operations_in_catalog() -> None:
    ids = {tool.operation_id for tool in TOOLS}
    assert "algebraic_number.add.compute" in ids
    assert "algebraic_number.multiply.compute" in ids
