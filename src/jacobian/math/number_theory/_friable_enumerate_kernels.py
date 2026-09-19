"""Exact bounded kernel for friable-family enumeration."""

from __future__ import annotations

from jacobian.math.number_theory._friable_enumerate_models import (
    plan_friable_enumerate,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def enumerate_friable(
    x: int | IntegerValue,
    y: int | IntegerValue,
) -> tuple[int, ...]:
    """Return the increasing tuple of positive y-friable integers at most x.

    The cutoff is inclusive. By convention, for positive x and y <= 1, only 1
    is friable. For x = 0, the family is empty.
    """

    x = _as_python_integer(x)
    y = _as_python_integer(y)
    if type(x) is not int or type(y) is not int:
        raise TypeError("friable-enumerate inputs must be integers")

    return plan_friable_enumerate(x, y)


def _as_python_integer(value: int | IntegerValue) -> int:
    """Return one admitted integer input as its Python integer value."""

    if isinstance(value, IntegerValue):
        return value.value
    return value


__all__ = ["enumerate_friable"]
