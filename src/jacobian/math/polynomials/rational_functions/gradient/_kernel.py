"""Admitted exact quotient differentiation and owner-canonical normalization."""

from typing import Any

from jacobian._execution import request_checkpoint
from jacobian.math.polynomials._conversions import sparse_rational_polynomial_from_sympy
from jacobian.math.polynomials.rational_functions.gradient._normalize_process import (
    cancel_fraction,
)
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
    numerator: Any,
    denominator: Any,
    variables: tuple[str, ...],
    *,
    deadline: float,
) -> RationalFunction:
    """Normalize an admitted pair once; preserve canonical field representation."""
    request_checkpoint("before rational gradient normalization")
    numerator, denominator = cancel_fraction(numerator, denominator, deadline=deadline)
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
