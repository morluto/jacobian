"""Admitted exact quotient differentiation and owner-canonical normalization."""

from typing import Any

from jacobian._execution import request_checkpoint
from jacobian.math.polynomials._conversions import sparse_rational_polynomial_from_sympy
from jacobian.math.polynomials.values import RationalFunction


def _differentiate_fraction(
    numerator: Any, denominator: Any, axis: int
) -> tuple[Any, Any]:
    """Return the raw quotient-rule pair after the caller admits its entire DAG.

    Inputs are exact SymPy QQ Poly values on identical ordered generators.
    Keeping normalization separate lets a tensor owner share its own complete
    arithmetic ledger, rather than invoking independent scalar admissions.
    """
    return (
        numerator.diff(axis) * denominator - numerator * denominator.diff(axis),
        denominator * denominator,
    )


def _normalize_fraction(
    numerator: Any, denominator: Any, variables: tuple[str, ...]
) -> RationalFunction:
    """Normalize an admitted pair once; preserve canonical field representation."""
    request_checkpoint("before rational gradient normalization")
    # Match the bound's guaranteed common-monomial presolve before GCD.
    # A uniform exponent shift preserves every coefficient and sparse term.
    if not numerator.is_zero:
        from sympy import Poly

        numerator_terms, denominator_terms = numerator.terms(), denominator.terms()
        common = tuple(
            min(
                min(exponents[axis] for exponents, _ in numerator_terms),
                min(exponents[axis] for exponents, _ in denominator_terms),
            )
            for axis in range(len(variables))
        )
        if any(common):

            def divide(value: Any) -> Any:
                return Poly.from_dict(
                    {
                        tuple(
                            e - c for e, c in zip(exponents, common, strict=True)
                        ): coefficient
                        for exponents, coefficient in value.terms()
                    },
                    value.gens,
                    domain=value.domain,
                )

            numerator, denominator = divide(numerator), divide(denominator)
    numerator, denominator = numerator.cancel(denominator, include=True)
    leading = denominator.LC()
    numerator = numerator.mul_ground(1 / leading)
    denominator = denominator.mul_ground(1 / leading)
    request_checkpoint("after rational gradient normalization")
    return RationalFunction._from_kernel(
        variables=variables,
        numerator=sparse_rational_polynomial_from_sympy(
            numerator, variables, maximum_terms=256
        ),
        denominator=sparse_rational_polynomial_from_sympy(
            denominator, variables, maximum_terms=256
        ),
    )
