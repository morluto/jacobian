"""Domain functions for Galois theory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy.combinatorics.permutations import PermutationGroup

from jacobian.math.galois_theory._models import (
    FrobeniusCycleRequest,
    FrobeniusCycleResult,
    GaloisFactorRequest,
    GaloisFactorResult,
    GaloisGroupRequest,
    GaloisGroupResult,
    SolvableRequest,
    SolvableResult,
)


def compute_galois_factor(request: GaloisFactorRequest) -> GaloisFactorResult:
    """Factor a polynomial over GF(p) using SymPy."""
    from sympy import GF, Poly, Symbol

    field = GF(request.field_order)
    x = Symbol("x")
    coeffs = list(request.coefficients)
    terms = sum(c * x**i for i, c in enumerate(coeffs))
    poly = Poly(terms, domain=field)
    _coeff, factor_polys = poly.factor_list()
    result_factors = []
    for factor_poly, _mult in factor_polys:
        coeff_list = [int(c) for c in factor_poly.all_coeffs()]
        result_factors.append(tuple(reversed(coeff_list)))

    factor_count = len(result_factors)
    is_irred = factor_count == 1
    return GaloisFactorResult(
        factors=tuple(result_factors),
        factor_count=factor_count,
        is_irreducible=is_irred,
    )


def compute_frobenius_cycle(request: FrobeniusCycleRequest) -> FrobeniusCycleResult:
    cycle_type = tuple(sorted(request.factorization_degrees, reverse=True))
    is_irred = len(cycle_type) == 1
    return FrobeniusCycleResult(
        cycle_type=cycle_type,
        degree=request.polynomial_degree,
        is_irreducible=is_irred,
    )


def _galois_group_from_coeffs(coeffs: tuple[int, ...]) -> PermutationGroup:
    """Return the SymPy permutation group for a polynomial over Q.

    Coefficients are in ascending order: coeffs[0] is the constant term,
    coeffs[-1] is the leading coefficient.  SymPy's ``Poly`` expects
    descending order, so we reverse.
    """
    from sympy import Poly, Symbol, galois_group

    x = Symbol("x")
    # coefficients[0] = constant, coefficients[-1] = leading
    # Poly expects highest-degree first
    descending = list(reversed(coeffs))
    poly = Poly(descending, x, domain="QQ")
    perm_group, _alt = galois_group(poly)
    return perm_group


def compute_galois_group(request: GaloisGroupRequest) -> GaloisGroupResult:
    """Compute the Galois group of a polynomial over Q."""
    perm_group = _galois_group_from_coeffs(request.coefficients)
    group_name = str(perm_group)
    order = int(perm_group.order())
    is_solvable = bool(perm_group.is_solvable)

    return GaloisGroupResult(
        group_name=group_name,
        order=order,
        degree=len(request.coefficients) - 1,
        is_solvable=is_solvable,
    )


def compute_solvable(request: SolvableRequest) -> SolvableResult:
    """Determine if a polynomial is solvable by radicals.

    A polynomial is solvable by radicals iff its Galois group is solvable.
    This is computed from the actual Galois group, not from the degree alone.
    """
    perm_group = _galois_group_from_coeffs(request.coefficients)
    is_solvable = bool(perm_group.is_solvable)
    return SolvableResult(solvable_by_radicals=is_solvable)
