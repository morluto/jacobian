"""SymPy-backed root isolation and comparison projection."""

from __future__ import annotations

import sympy
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
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
    try:
        source_coefficients = request.normalized_integer_coefficients()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("polynomial",), code=exc.type, message=exc.message()
        ) from exc
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
        algebraic_value = RealAlgebraicValue._from_admitted_polynomial(
            polynomial=tuple(factor_coefficients),
            real_root_index=left_roots,
        )
        roots.append(
            RootIsolationEntry(
                isolating_interval=(
                    CanonicalRational(
                        num=int(sympy.Rational(lower).p),
                        den=int(sympy.Rational(lower).q),
                    ),
                    CanonicalRational(
                        num=int(sympy.Rational(upper).p),
                        den=int(sympy.Rational(upper).q),
                    ),
                ),
                multiplicity=source_multiplicity,
                algebraic_value=algebraic_value,
            )
        )
    return RootIsolationResult._from_kernel(
        source_polynomial=request.polynomial,
        roots=tuple(roots),
    )


def compute_algebraic_compare(
    request: AlgebraicCompareRequest,
) -> RealAlgebraicOrderValue:
    return compare_algebraic(request.left, request.right)
