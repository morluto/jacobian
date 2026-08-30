"""Cyclic sumset representation profile kernel."""

from __future__ import annotations

from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.cyclic_sumset_profile._models import (
    MAX_CYCLIC_SUMSET_PAIRS,
    CyclicSumsetEntry,
    CyclicSumsetResult,
    _result_wire_bytes,
)

__all__ = ["compute_cyclic_sumset_profile"]


def compute_cyclic_sumset_profile(
    modulus: int,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> CyclicSumsetResult:
    """Return the complete cyclic representation function r_{A+B}(c)."""
    if modulus <= 0:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="cyclic_sumset.positive_modulus",
            message="cyclic sumset modulus must be positive",
        )
    if len(left) * len(right) > MAX_CYCLIC_SUMSET_PAIRS:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="cyclic_sumset.pair_work_exceeded",
            message="cyclic sumset exceeds the 100000-pair work bound",
        )
    if any(not 0 <= value < modulus for value in (*left, *right)):
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="cyclic_sumset.canonical_residue",
            message="cyclic sumset operands must be canonical residues modulo modulus",
        )
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="cyclic_sumset.duplicate_operand",
            message="cyclic sumset operands must contain distinct residues",
        )
    if _result_wire_bytes(modulus, left, right) > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="cyclic_sumset.result_bytes_exceeded",
            message="cyclic sumset profile exceeds the canonical output-byte limit",
        )
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
