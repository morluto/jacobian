"""Cyclic prefix-sum residue profile kernel."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian._exact import CanonicalInteger
from jacobian.canonical import (
    CanonicalizationError,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.cyclic_prefix_sum._models import (
    MAX_MODULUS_DIGITS,
    MAX_RESULT_BYTES,
    MAX_SEQUENCE_LENGTH,
    CyclicPrefixSumResidueProfileResult,
    PrefixSumResidueRow,
)
from jacobian.math.combinatorics.additive.values import IndexedIntegerSequence

__all__ = ["compute_cyclic_prefix_sum_residue_profile"]

MAX_WORK_UNITS = 50_000_000


@dataclass(frozen=True, slots=True)
class _AdmissionPlan:
    sequence: tuple[int, ...]
    modulus: int


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _admit(
    sequence: IndexedIntegerSequence,
    modulus: CanonicalInteger,
) -> _AdmissionPlan:
    """Validate the complete native and MCP execution envelope once."""
    if not isinstance(sequence, IndexedIntegerSequence):
        _reject(
            ("sequence",),
            "cyclic_prefix_sum.sequence_type",
            "sequence must be an indexed integer sequence",
        )
    if type(modulus) is not str:
        _reject(
            ("modulus",),
            "cyclic_prefix_sum.modulus_type",
            "modulus must be a canonical integer string",
        )
    modulus_digits = len(modulus.lstrip("-"))
    if modulus_digits > MAX_MODULUS_DIGITS:
        _reject(
            ("modulus",),
            "cyclic_prefix_sum.modulus_digits",
            f"modulus may contain at most {MAX_MODULUS_DIGITS} digits",
        )
    try:
        modulus_value = parse_canonical_integer(modulus)
    except CanonicalizationError:
        _reject(
            ("modulus",),
            "cyclic_prefix_sum.modulus_format",
            "modulus must be a canonical integer string",
        )
    if modulus_value <= 0:
        _reject(
            ("modulus",),
            "cyclic_prefix_sum.modulus_domain",
            "modulus must be positive",
        )

    item_count = len(sequence.items)
    if item_count > MAX_SEQUENCE_LENGTH:
        _reject(
            ("sequence",),
            "cyclic_prefix_sum.sequence_length",
            f"sequence may contain at most {MAX_SEQUENCE_LENGTH:,} items",
        )
    maximum_item_digits = max(
        (len(item.lstrip("-")) for item in sequence.items),
        default=1,
    )
    work = item_count * max(maximum_item_digits, modulus_digits)
    if work > MAX_WORK_UNITS:
        _reject(
            ("sequence",),
            "cyclic_prefix_sum.work_bound",
            "prefix-sum modular arithmetic exceeds the admitted work bound",
        )

    row_count = min(item_count, modulus_value)
    position_digits = len(str(max(item_count, 1)))
    result_bytes = (
        256
        + modulus_digits
        + row_count * (modulus_digits + 32)
        + item_count * (position_digits + 2)
    )
    if result_bytes > MAX_RESULT_BYTES:
        _reject(
            ("sequence",),
            "cyclic_prefix_sum.result_bound",
            "the exact residue profile exceeds the canonical output bound",
        )
    return _AdmissionPlan(sequence=sequence.as_int_tuple(), modulus=modulus_value)


def compute_cyclic_prefix_sum_residue_profile(
    sequence: IndexedIntegerSequence,
    modulus: CanonicalInteger,
) -> CyclicPrefixSumResidueProfileResult:
    """Return the complete partition of prefix positions by residue.

    For each prefix position k (1-indexed), compute the prefix sum
    S_k = a_1 + ... + a_k mod m and group positions by their residue.
    """
    plan = _admit(sequence, modulus)
    residue_to_positions: dict[int, list[int]] = {}
    running = 0
    for k, value in enumerate(plan.sequence, start=1):
        running = (running + value) % plan.modulus
        if running not in residue_to_positions:
            residue_to_positions[running] = []
        residue_to_positions[running].append(k)

    rows = [
        PrefixSumResidueRow(
            residue=format_canonical_integer(res), positions=tuple(positions)
        )
        for res, positions in sorted(residue_to_positions.items())
    ]
    return CyclicPrefixSumResidueProfileResult(
        modulus=format_canonical_integer(plan.modulus),
        rows=tuple(rows),
    )
