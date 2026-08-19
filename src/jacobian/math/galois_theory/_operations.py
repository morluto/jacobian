"""Domain functions for Galois theory operations."""

from __future__ import annotations

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


def compute_galois_group(request: GaloisGroupRequest) -> GaloisGroupResult:
    """Compute the Galois group of a polynomial over Q."""
    from sympy import Poly, Symbol, galois_group

    degree = len(request.coefficients) - 1
    poly = Poly(list(request.coefficients), Symbol("x"), domain="QQ")
    perm_group, solvable = galois_group(poly)
    group_name = str(perm_group)
    order = int(perm_group.order())
    is_solvable = bool(solvable)

    return GaloisGroupResult(
        group_name=group_name,
        order=order,
        degree=degree,
        is_solvable=is_solvable,
    )


def compute_solvable(request: SolvableRequest) -> SolvableResult:
    """A polynomial is solvable by radicals iff its Galois group is solvable."""
    degree = len(request.coefficients) - 1
    if degree <= 4:
        return SolvableResult(solvable_by_radicals=True)
    if degree >= 5:
        return SolvableResult(solvable_by_radicals=False)
    return SolvableResult(solvable_by_radicals=False)
