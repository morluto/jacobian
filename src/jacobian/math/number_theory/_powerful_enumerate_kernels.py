"""Exact kernel for bounded powerful-number enumeration.

A powerful integer n > 1 has the canonical square-cube representation n = a^2 * b^3
where b is squarefree. This representation is unique, so we enumerate:
  - squarefree b with b^3 <= cutoff,
  - a with a^2 * b^3 <= cutoff,
and deduplicate/sort under a complete interval envelope.
"""

from __future__ import annotations


def _is_squarefree(n: int) -> bool:
    """Return True if n is squarefree (1 is squarefree)."""
    if n <= 0:
        return False
    if n <= 1:
        return True
    d = 2
    while d * d <= n:
        if n % d == 0:
            n //= d
            if n % d == 0:
                return False
        d += 1
    return True


def enumerate_powerful(cutoff: int) -> list[int]:
    """Return every powerful integer in [1, cutoff] exactly once, sorted.

    Uses the canonical square-cube representation n = a^2 * b^3
    with b squarefree. The uniqueness of this representation guarantees
    each powerful integer is generated exactly once.
    """
    family: set[int] = set()

    # b iterates over squarefree integers with b^3 <= cutoff.
    b = 0
    while True:
        b += 1
        b3 = b * b * b
        if b3 > cutoff:
            break
        if not _is_squarefree(b):
            continue
        # a iterates over positive integers with a^2 * b^3 <= cutoff.
        a = 1
        while True:
            a2 = a * a
            if a2 * b3 > cutoff:
                break
            family.add(a2 * b3)
            a += 1

    return sorted(family)
