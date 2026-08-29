"""Domain functions for Galois theory operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError

if TYPE_CHECKING:
    from sympy.combinatorics.perm_groups import PermutationGroup

from jacobian.math.number_theory.galois._models import (
    FiniteFieldFactor,
    FinitePermutationGroup,
    FrobeniusCycleResult,
    GaloisFactorResult,
    GaloisGroupResult,
    SolvableResult,
    _require_prime,
    _supported_galois_polynomial,
)


def _admit(operation: Callable[[], None], *, location: tuple[str | int, ...]) -> None:
    try:
        operation()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


def _admit_factor(field_order: int, coefficients: tuple[int, ...]) -> None:
    _require_prime(field_order)
    if any(not 0 <= coefficient < field_order for coefficient in coefficients):
        raise PydanticCustomError(
            "galois_theory.coefficients_not_canonical",
            "coefficients must be canonical field residues",
        )
    if not coefficients or coefficients[-1] == 0:
        raise PydanticCustomError(
            "galois_theory.polynomial_zero",
            "factorization requires a nonzero polynomial with canonical degree",
        )


def _admit_frobenius(
    field_order: int,
    polynomial_degree: int,
    factorization_degrees: tuple[int, ...],
) -> None:
    from collections import Counter

    from sympy import divisors, mobius

    _require_prime(field_order)
    if sum(factorization_degrees) != polynomial_degree:
        raise PydanticCustomError(
            "galois_theory.partition_degree_mismatch",
            "factorization degrees must sum to polynomial degree",
        )
    for degree, count in Counter(factorization_degrees).items():
        available = (
            sum(
                int(mobius(divisor)) * field_order ** (degree // divisor)
                for divisor in divisors(degree)
            )
            // degree
        )
        if count > available:
            raise PydanticCustomError(
                "galois_theory.partition_unrealizable",
                "factorization pattern exceeds the available distinct "
                f"degree-{degree} irreducible factors over the field",
            )


def galois_factor(
    field_order: int, coefficients: tuple[int, ...]
) -> GaloisFactorResult:
    """Factor a polynomial over GF(p) using SymPy."""
    _admit(
        lambda: _admit_factor(field_order, coefficients),
        location=("field_order", "coefficients"),
    )
    from sympy import GF, Poly, Symbol

    field = GF(field_order)
    x = Symbol("x")
    coeffs = list(coefficients)
    terms = sum(c * x**i for i, c in enumerate(coeffs))
    poly = Poly(terms, domain=field)
    unit, factor_polys = poly.factor_list()
    result_factors = tuple(
        FiniteFieldFactor(
            coefficients=tuple(
                int(coefficient) % field_order
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
        and len(result_factors[0].coefficients) == len(coefficients)
    )
    return GaloisFactorResult._from_kernel(
        field_order=field_order,
        source_coefficients=coefficients,
        unit=int(unit) % field_order,
        factors=result_factors,
        distinct_factor_count=len(result_factors),
        factor_count=factor_count,
        is_irreducible=is_irred,
    )


def frobenius_cycle(
    field_order: int,
    polynomial_degree: int,
    factorization_degrees: tuple[int, ...],
) -> FrobeniusCycleResult:
    _admit(
        lambda: _admit_frobenius(field_order, polynomial_degree, factorization_degrees),
        location=("field_order", "factorization_degrees"),
    )
    cycle_type = tuple(sorted(factorization_degrees, reverse=True))
    is_irred = cycle_type == (polynomial_degree,)
    return FrobeniusCycleResult(
        cycle_type=cycle_type,
        degree=polynomial_degree,
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


def galois_group(coefficients: tuple[int, ...]) -> GaloisGroupResult:
    """Compute the Galois group of a polynomial over Q."""
    _admit(
        lambda: _supported_galois_polynomial(coefficients),
        location=("coefficients",),
    )
    perm_group = _galois_group_from_coeffs(coefficients)
    group_name = str(perm_group)
    order = int(perm_group.order())
    is_solvable = bool(perm_group.is_solvable)

    return GaloisGroupResult._from_kernel(
        group=_wire_group(perm_group, len(coefficients) - 1),
        group_name=group_name,
        order=order,
        degree=len(coefficients) - 1,
        is_solvable=is_solvable,
    )


def solvable(coefficients: tuple[int, ...]) -> SolvableResult:
    """Determine if a polynomial is solvable by radicals.

    A polynomial is solvable by radicals iff its Galois group is solvable.
    This is computed from the actual Galois group, not from the degree alone.
    """
    _admit(
        lambda: _supported_galois_polynomial(coefficients),
        location=("coefficients",),
    )
    perm_group = _galois_group_from_coeffs(coefficients)
    is_solvable = bool(perm_group.is_solvable)
    return SolvableResult._from_kernel(
        solvable_by_radicals=is_solvable,
        group=_wire_group(perm_group, len(coefficients) - 1),
    )


__all__ = [
    "frobenius_cycle",
    "galois_factor",
    "galois_group",
    "solvable",
]
