"""Tests for Galois theory operations."""

from jacobian.math.galois_theory._models import (
    FrobeniusCycleRequest,
    GaloisFactorRequest,
    SolvableRequest,
)
from jacobian.math.galois_theory._operations import (
    compute_frobenius_cycle,
    compute_galois_factor,
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


def test_solvable_cubic() -> None:
    request = SolvableRequest(coefficients=(-2, 0, 0, 1))
    result = compute_solvable(request)
    assert result.solvable_by_radicals is True


def test_solvable_quintic() -> None:
    request = SolvableRequest(coefficients=(-1, 0, 0, 0, 0, 1))
    result = compute_solvable(request)
    assert result.solvable_by_radicals is False
