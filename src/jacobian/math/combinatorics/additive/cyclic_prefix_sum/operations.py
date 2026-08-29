"""Cyclic prefix-sum residue profile kernel."""

from __future__ import annotations

from jacobian.math.combinatorics.additive.cyclic_prefix_sum._models import (
    CyclicPrefixSumResidueProfileResult,
    PrefixSumResidueRow,
)

__all__ = ["compute_cyclic_prefix_sum_residue_profile"]


def compute_cyclic_prefix_sum_residue_profile(
    sequence: tuple[int, ...],
    modulus: int,
) -> CyclicPrefixSumResidueProfileResult:
    """Return the complete partition of prefix positions by residue.

    For each prefix position k (1-indexed), compute the prefix sum
    S_k = a_1 + ... + a_k mod m and group positions by their residue.
    """
    residue_to_positions: dict[int, list[int]] = {}
    running = 0
    for k, value in enumerate(sequence, start=1):
        running = (running + value) % modulus
        if running not in residue_to_positions:
            residue_to_positions[running] = []
        residue_to_positions[running].append(k)

    rows = [
        PrefixSumResidueRow(residue=res, positions=tuple(positions))
        for res, positions in sorted(residue_to_positions.items())
    ]
    return CyclicPrefixSumResidueProfileResult(
        modulus=modulus,
        rows=tuple(rows),
    )
