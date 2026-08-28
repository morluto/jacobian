"""Canonical exact values owned by the arithmetic domain."""

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

__all__ = ["IntegerValue"]


class IntegerValue(StrictModel):
    """One canonical exact integer, usable as either an input or a result."""

    value: CanonicalInteger
