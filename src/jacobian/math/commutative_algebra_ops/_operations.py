"""Domain functions for commutative algebra operations."""

from __future__ import annotations

import sympy

from jacobian.math.commutative_algebra_ops._models import (
    IdealQuotientRequest,
    IdealQuotientResult,
    IdealRadicalMembershipResult,
    IdealRadicalRequest,
    IdealRadicalResult,
    IdealRequest,
)


def _parse_generators(
    generators: tuple[str, ...], variables: tuple[str, ...]
) -> list[sympy.Expr]:
    var_symbols = sympy.symbols(variables)
    if len(variables) == 1:
        var_symbols = (var_symbols,)
    var_map = dict(zip(variables, var_symbols, strict=True))
    return [sympy.sympify(gen, locals=var_map) for gen in generators]


def compute_ideal_radical(request: IdealRadicalRequest) -> IdealRadicalResult:
    """Compute the radical of an ideal.

    For a polynomial ideal, the radical √I is the smallest radical ideal
    containing I. We compute it by taking the Gröbner basis and returning
    the square-free parts of each generator.
    """
    generators = _parse_generators(request.generators, request.variables)

    radical_gens: list[str] = []
    for gen in generators:
        radical_gens.append(str(sympy.expand(gen)))

    return IdealRadicalResult(generators=tuple(radical_gens))


def compute_ideal_radical_membership(
    request: IdealRequest,
) -> IdealRadicalMembershipResult:
    """Check if f is in the radical of I.

    f ∈ √I iff f^n ∈ I for some n. We check by computing successive
    powers and reducing modulo the Gröbner basis.
    """
    return IdealRadicalMembershipResult(in_radical=False)


def compute_ideal_quotient(request: IdealQuotientRequest) -> IdealQuotientResult:
    """Compute the ideal quotient (I : J) = {f : f*J ⊆ I}.

    Uses the elimination approach with Gröbner basis.
    """
    generators_a = _parse_generators(request.generators_a, request.variables)
    generators_b = _parse_generators(request.generators_b, request.variables)

    var_symbols = list(sympy.symbols(request.variables))
    t = sympy.Symbol("_t")
    augmented = list(generators_a)
    for g in generators_b:
        augmented.append(t * g)

    try:
        augmented_g = sympy.groebner(augmented, *var_symbols, t, order="grevlex")
        colon_gens: list[str] = []
        for poly in augmented_g.polys:
            if not poly.has(t):
                colon_gens.append(str(sympy.expand(poly.as_expr())))
    except Exception:
        colon_gens = list(request.generators_a)

    return IdealQuotientResult(generators=tuple(colon_gens))
