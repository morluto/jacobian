"""Shared raw and canonical work bounds for planar rigidity profiles."""

from __future__ import annotations

MAX_FRAMEWORK_COORDINATE_WORK = 100_000_000
"""Decimal-limb and rational-difference work admitted before matrix expansion."""


def rational_parse_work(components: tuple[str, str]) -> int:
    """Bound base-10**9 multiply-add visits for two rational components."""

    work = 0
    for component in components:
        digits = len(component.lstrip("-"))
        limbs = (digits + 8) // 9
        work += limbs * (limbs + 1)
    return work


def rational_height(components: tuple[str, str]) -> int:
    """Return the largest canonical decimal component width."""

    return max(len(component.lstrip("-")) for component in components)


def difference_work(left: tuple[str, str], right: tuple[str, str]) -> int:
    """Price subtraction plus both signed canonical sparse-entry projections."""

    height = max(rational_height(left), rational_height(right))
    # A quadratic height charge conservatively covers the two cross-products,
    # subtraction, reduction, and decimal projection of both signed entries.
    return 32 * height * height + 32 * height


__all__ = [
    "MAX_FRAMEWORK_COORDINATE_WORK",
    "difference_work",
    "rational_height",
    "rational_parse_work",
]
