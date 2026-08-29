"""Cyclic sumset representation profile kernel."""

from __future__ import annotations

from jacobian.math.combinatorics.additive.cyclic_sumset_profile._models import (
    CyclicSumsetEntry,
    CyclicSumsetResult,
)

__all__ = ["compute_cyclic_sumset_profile"]


def compute_cyclic_sumset_profile(
    modulus: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> CyclicSumsetResult:
    """Return the complete cyclic representation function r_{A+B}(c)."""
    counts: dict[int, int] = {}
    for a in left:
        for b in right:
            c = (a + b) % modulus
            counts[c] = counts.get(c, 0) + 1

    entries = tuple(
        CyclicSumsetEntry(residue=r, count=ct) for r, ct in sorted(counts.items())
    )

    return CyclicSumsetResult(
        modulus=modulus,
        left=left,
        right=right,
        entries=entries,
        support_cardinality=len(counts),
    )
