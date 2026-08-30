from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.collatz_wielandt.operations import (
    compute_collatz_wielandt_profile,
)


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def test_identity() -> None:
    matrix = ((_cr(1), _cr(0)), (_cr(0), _cr(1)))
    vector = (_cr(1), _cr(1))
    result = compute_collatz_wielandt_profile(matrix, vector)
    assert result.quotients[0].as_fraction() == Fraction(1)
    assert result.quotients[1].as_fraction() == Fraction(1)
    assert result.max_quotient.as_fraction() == Fraction(1)


def test_2x2() -> None:
    matrix = ((_cr(2), _cr(1)), (_cr(0), _cr(3)))
    vector = (_cr(1), _cr(1))
    result = compute_collatz_wielandt_profile(matrix, vector)
    assert result.quotients[0].as_fraction() == Fraction(3)
    assert result.quotients[1].as_fraction() == Fraction(3)


def test_non_uniform_vector() -> None:
    matrix = ((_cr(1), _cr(2)), (_cr(3), _cr(4)))
    vector = (_cr(1), _cr(2))
    result = compute_collatz_wielandt_profile(matrix, vector)
    assert result.quotients[0].as_fraction() == Fraction(5)
    assert result.quotients[1].as_fraction() == Fraction(11, 2)
    assert result.max_quotient.as_fraction() == Fraction(11, 2)


def test_result_preserves_source() -> None:
    matrix = ((_cr(1),),)
    vector = (_cr(1),)
    result = compute_collatz_wielandt_profile(matrix, vector)
    assert result.matrix == matrix
    assert result.vector == vector


def test_negative_matrix_entry_is_rejected_before_quotients() -> None:
    with pytest.raises(OperationDomainValidationError, match="nonnegative matrix"):
        compute_collatz_wielandt_profile(((_cr(-1),),), (_cr(1),))


def test_nonpositive_vector_is_rejected_through_the_domain_boundary() -> None:
    with pytest.raises(OperationDomainValidationError, match="positive vector"):
        compute_collatz_wielandt_profile(((_cr(1),),), (_cr(0),))


def test_derived_quotient_must_fit_the_rational_carrier() -> None:
    denominator = 10**20_000
    matrix = (
        (_cr(1, denominator), _cr(1, denominator - 1)),
        (_cr(0), _cr(0)),
    )
    vector = (_cr(1), _cr(1))

    with pytest.raises(OperationDomainValidationError, match=r"derived.*quotient"):
        compute_collatz_wielandt_profile(matrix, vector)


def test_aggregate_result_bound_is_checked_before_arithmetic() -> None:
    denominator = "1" + "0" * 32_767
    value = CanonicalRational(num="1", den=denominator)
    matrix = tuple(tuple(value for _ in range(17)) for _ in range(17))
    vector = tuple(value for _ in range(17))
    with pytest.raises(OperationDomainValidationError, match="profile exceeds"):
        compute_collatz_wielandt_profile(matrix, vector)
