"""Exact SymPy root-isolation primitives shared by mathematical owners."""

from __future__ import annotations

from typing import Any

__all__ = ["strict_root_count"]


def strict_root_count(poly: Any, lower: Any, upper: Any) -> int:
    """Count roots strictly between endpoints, or at a singleton endpoint.

    SymPy's interval convention includes roots at the endpoints. The
    root-isolation owners use open intervals for irrational roots and a
    singleton for a rational root, so remove endpoint roots explicitly.
    """

    if lower == upper:
        return int(poly.eval(lower) == 0)
    count = int(poly.count_roots(lower, upper))
    if poly.eval(lower) == 0:
        count -= 1
    if poly.eval(upper) == 0:
        count -= 1
    return count
