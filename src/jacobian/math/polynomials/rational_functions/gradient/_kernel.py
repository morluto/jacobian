"""Admitted exact quotient differentiation and owner-canonical normalization."""

from jacobian._execution import request_checkpoint
from jacobian.math.polynomials.rational_functions.gradient._normalize_process import (
    normalize_partial,
)
from jacobian.math.polynomials.values import RationalFunction


def _normalize_fraction(
    source: RationalFunction,
    axis: int,
    *,
    deadline: float,
) -> RationalFunction:
    """Normalize one admitted partial; preserve canonical field representation."""
    request_checkpoint("before rational gradient normalization")
    result = normalize_partial(source, axis, deadline=deadline)
    request_checkpoint("after rational gradient normalization")
    return result
