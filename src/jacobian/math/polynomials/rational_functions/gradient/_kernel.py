"""Admitted exact quotient differentiation and owner-canonical normalization."""

from typing import Any

from jacobian._execution import request_checkpoint
from jacobian.math.polynomials.rational_functions.gradient._gcd_process import (
    differentiate_admitted_fraction,
    normalize_admitted_fraction,
)
from jacobian.math.polynomials.values import RationalFunction


def _differentiate_fraction(
    source: RationalFunction,
    axis: int,
    factor_records: tuple[list[Any], ...] = (),
) -> RationalFunction:
    """Differentiate an admitted fraction in the killable GCD worker."""
    request_checkpoint("before rational gradient differentiation")
    result = differentiate_admitted_fraction(source, axis, factor_records)
    request_checkpoint("after rational gradient differentiation")
    return result


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
