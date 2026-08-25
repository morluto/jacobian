"""Exact predicates over canonical integer values."""

from math import isqrt


def is_square_free(value: int) -> bool:
    """Return whether no square greater than one divides ``value``."""

    return all(value % (divisor * divisor) for divisor in range(2, isqrt(value) + 1))


__all__ = ["is_square_free"]
