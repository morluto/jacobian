"""Admitted exact quotient differentiation and owner-canonical normalization."""

from typing import Any

from jacobian._execution import request_checkpoint
from jacobian.math.polynomials.rational_functions.gradient._gcd_process import (
    normalize_admitted_fraction,
)


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
    factor_records: tuple[list[Any], ...] = (),
) -> Any:
    """Normalize an admitted pair; reuse any retained derivative factor."""
    request_checkpoint("before rational gradient normalization")
    result = normalize_admitted_fraction(
        numerator, denominator, variables, factor_records
    )
    request_checkpoint("after rational gradient normalization")
    return result
