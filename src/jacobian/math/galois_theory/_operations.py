"""Domain functions for Galois theory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy.combinatorics.perm_groups import PermutationGroup

from jacobian.math.galois_theory._models import (
    FiniteFieldFactor,
    FinitePermutationGroup,
    FrobeniusCycleRequest,
    FrobeniusCycleResult,
    GaloisFactorRequest,
    GaloisFactorResult,
    GaloisGroupRequest,
    GaloisGroupResult,
    SolvableRequest,
    SolvableResult,
    _permutation_group_properties,
)


def compute_galois_factor(request: GaloisFactorRequest) -> GaloisFactorResult:
    """Factor a polynomial over GF(p) using SymPy."""
    from sympy import GF, Poly, Symbol

    field = GF(request.field_order)
    x = Symbol("x")
    coeffs = list(request.coefficients)
    terms = sum(c * x**i for i, c in enumerate(coeffs))
    poly = Poly(terms, domain=field)
    unit, factor_polys = poly.factor_list()
    result_factors = tuple(
        FiniteFieldFactor(
            coefficients=tuple(
                int(coefficient) % request.field_order
                for coefficient in reversed(factor_poly.all_coeffs())
            ),
            multiplicity=int(multiplicity),
        )
        for factor_poly, multiplicity in factor_polys
    )
    factor_count = sum(factor.multiplicity for factor in result_factors)
    is_irred = (
        len(result_factors) == 1
        and result_factors[0].multiplicity == 1
        and len(result_factors[0].coefficients) == len(request.coefficients)
    )
    return GaloisFactorResult._from_kernel(
        field_order=request.field_order,
        source_coefficients=request.coefficients,
        unit=int(unit) % request.field_order,
        factors=result_factors,
        distinct_factor_count=len(result_factors),
        factor_count=factor_count,
        is_irreducible=is_irred,
    )


def verify_galois_factor_result(result: GaloisFactorResult) -> bool:
    """Replay bounded factor irreducibility for a supplied factorization claim."""

    from sympy import GF, Poly, Symbol

    x = Symbol("x")
    return result.is_irreducible == (
        len(result.factors) == 1
        and result.factors[0].multiplicity == 1
        and len(result.factors[0].coefficients) == len(result.source_coefficients)
    ) and all(
        Poly(
            list(reversed(factor.coefficients)), x, domain=GF(result.field_order)
        ).is_irreducible
        for factor in result.factors
    )


def compute_frobenius_cycle(request: FrobeniusCycleRequest) -> FrobeniusCycleResult:
    cycle_type = tuple(sorted(request.factorization_degrees, reverse=True))
    is_irred = cycle_type == (request.polynomial_degree,)
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


def _wire_group(perm_group: PermutationGroup, degree: int) -> FinitePermutationGroup:
    """Project a SymPy group onto a complete explicit root axis."""

    return FinitePermutationGroup(
        root_axis=tuple(f"root_{index}" for index in range(degree)),
        generators=tuple(
            tuple(int(generator(index)) for index in range(degree))
            for generator in perm_group.generators
        ),
    )


def compute_galois_group(request: GaloisGroupRequest) -> GaloisGroupResult:
    """Compute the Galois group of a polynomial over Q."""
    perm_group = _galois_group_from_coeffs(request.coefficients)
    group_name = str(perm_group)
    order = int(perm_group.order())
    is_solvable = bool(perm_group.is_solvable)

    return GaloisGroupResult._from_kernel(
        group=_wire_group(perm_group, len(request.coefficients) - 1),
        group_name=group_name,
        order=order,
        degree=len(request.coefficients) - 1,
        is_solvable=is_solvable,
    )


def verify_galois_group_result(result: GaloisGroupResult) -> bool:
    """Replay bounded group properties for an independently supplied claim."""

    order, is_solvable = _permutation_group_properties(result.group)
    return (
        result.degree == len(result.group.root_axis)
        and result.order == order
        and result.is_solvable == is_solvable
    )


def compute_solvable(request: SolvableRequest) -> SolvableResult:
    """Determine if a polynomial is solvable by radicals.

    A polynomial is solvable by radicals iff its Galois group is solvable.
    This is computed from the actual Galois group, not from the degree alone.
    """
    perm_group = _galois_group_from_coeffs(request.coefficients)
    is_solvable = bool(perm_group.is_solvable)
    return SolvableResult._from_kernel(
        solvable_by_radicals=is_solvable,
        group=_wire_group(perm_group, len(request.coefficients) - 1),
    )


def verify_solvable_result(result: SolvableResult) -> bool:
    """Replay bounded group solvability for an independently supplied claim."""

    _, is_solvable = _permutation_group_properties(result.group)
    return result.solvable_by_radicals == is_solvable
