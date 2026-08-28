"""Domain-owned root isolation operations."""

from __future__ import annotations

import sympy

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.number_theory.algebraic_numbers.real import (
    RealAlgebraicOrderValue,
    RealAlgebraicValue,
)
from jacobian.math.number_theory.algebraic_numbers.root_isolation import (
    compare_algebraic,
)
from jacobian.math.number_theory.algebraic_numbers.root_isolation._models import (
    AlgebraicCompareRequest,
    RootIsolationEntry,
    RootIsolationResult,
    UnivariatePolynomialRequest,
)


def compute_root_isolation(request: UnivariatePolynomialRequest) -> RootIsolationResult:
    source_coefficients = request.normalized_integer_coefficients()
    variable = sympy.Symbol("x")
    source = sympy.Poly.from_list(source_coefficients, gens=variable, domain=sympy.ZZ)
    _unit, factorization = sympy.factor_list(source.as_expr(), variable)
    factors = tuple(
        (sympy.Poly(factor, variable, domain=sympy.ZZ), multiplicity)
        for factor, multiplicity in factorization
    )

    roots: list[RootIsolationEntry] = []
    for (lower, upper), _multiplicity in source.intervals():
        owning_factor, source_multiplicity = next(
            (factor, exponent)
            for factor, exponent in factors
            if strict_root_count(factor, lower, upper) == 1
        )
        factor_coefficients = tuple(int(value) for value in owning_factor.all_coeffs())
        left_roots = int(owning_factor.count_roots(-sympy.oo, lower))
        if owning_factor.eval(lower) == 0:
            left_roots -= 1
        algebraic_value = RealAlgebraicValue(
            polynomial=tuple(
                format_canonical_integer(coefficient)
                for coefficient in factor_coefficients
            ),
            real_root_index=left_roots,
        )
        roots.append(
            RootIsolationEntry(
                isolating_interval=(
                    CanonicalRational(
                        num=format_canonical_integer(sympy.Rational(lower).p),
                        den=format_canonical_integer(sympy.Rational(lower).q),
                    ),
                    CanonicalRational(
                        num=format_canonical_integer(sympy.Rational(upper).p),
                        den=format_canonical_integer(sympy.Rational(upper).q),
                    ),
                ),
                multiplicity=source_multiplicity,
                algebraic_value=algebraic_value,
            )
        )
    return RootIsolationResult._from_kernel(
        source_coefficients_descending=source_coefficients,
        roots=tuple(roots),
    )


def compute_algebraic_compare(
    request: AlgebraicCompareRequest,
) -> RealAlgebraicOrderValue:
    return compare_algebraic(request.left, request.right)
