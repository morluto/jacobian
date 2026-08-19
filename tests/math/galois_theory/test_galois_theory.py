"""Tests for Galois theory operations."""

from jacobian.math.galois_theory._models import (
    FrobeniusCycleRequest,
    GaloisFactorRequest,
    GaloisGroupRequest,
    SolvableRequest,
)
from jacobian.math.galois_theory._operations import (
    compute_frobenius_cycle,
    compute_galois_factor,
    compute_galois_group,
    compute_solvable,
)
from jacobian.math.galois_theory._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "polynomial.galois.factor_mod_p.compute",
        "polynomial.galois.frobenius_cycle.compute",
        "polynomial.galois_group.compute",
        "polynomial.solvable_by_radicals.decide",
    }


def test_factor_x2_plus_1_over_f5() -> None:
    request = GaloisFactorRequest(field_order=5, coefficients=(1, 0, 1))
    result = compute_galois_factor(request)
    assert result.factor_count == 2
    assert not result.is_irreducible


def test_frobenius_cycle_irreducible() -> None:
    request = FrobeniusCycleRequest(
        field_order=3, polynomial_degree=2, factorization_degrees=(2,)
    )
    result = compute_frobenius_cycle(request)
    assert result.cycle_type == (2,)
    assert result.is_irreducible is True


def test_frobenius_cycle_split() -> None:
    request = FrobeniusCycleRequest(
        field_order=5, polynomial_degree=2, factorization_degrees=(1, 1)
    )
    result = compute_frobenius_cycle(request)
    assert result.cycle_type == (1, 1)
    assert result.is_irreducible is False


# --- Issue 1: galois_group() second return value is not solvability ---


def test_galois_group_solvable_quintic() -> None:
    """x^5 - 2 has a solvable Galois group (Frobenius group of order 20)."""
    # coefficients in ascending order: (-2, 0, 0, 0, 0, 1) = -2 + x^5
    request = GaloisGroupRequest(coefficients=(-2, 0, 0, 0, 0, 1))
    result = compute_galois_group(request)
    assert result.degree == 5
    assert result.is_solvable is True
    assert result.order == 20


def test_galois_group_unsolvable_quintic() -> None:
    """x^5 - x - 1 has Galois group S_5 (not solvable)."""
    # coefficients in ascending order: (-1, -1, 0, 0, 0, 1) = -1 - x + x^5
    request = GaloisGroupRequest(coefficients=(-1, -1, 0, 0, 0, 1))
    result = compute_galois_group(request)
    assert result.degree == 5
    assert result.is_solvable is False
    assert result.order == 120  # |S_5| = 120


# --- Issue 2: radical solvability must use actual Galois group, not degree ---


def test_solvable_cubic() -> None:
    request = SolvableRequest(coefficients=(-2, 0, 0, 1))
    result = compute_solvable(request)
    assert result.solvable_by_radicals is True


def test_solvable_quintic_x5_minus_2() -> None:
    """x^5 - 2 is solvable by radicals even though degree is 5."""
    request = SolvableRequest(coefficients=(-2, 0, 0, 0, 0, 1))
    result = compute_solvable(request)
    assert result.solvable_by_radicals is True


def test_not_solvable_quintic_s5() -> None:
    """x^5 - x - 1 has Galois group S_5, not solvable by radicals."""
    request = SolvableRequest(coefficients=(-1, -1, 0, 0, 0, 1))
    result = compute_solvable(request)
    assert result.solvable_by_radicals is False
