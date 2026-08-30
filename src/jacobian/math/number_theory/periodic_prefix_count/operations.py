"""Periodic congruence union prefix count kernel."""

from __future__ import annotations

from typing import NoReturn

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic_kernel import (
    _ExecutionPlan,
    _merge_congruences,
    _sparse_union,
    _union_mask,
    require_admitted_periodic_source,
)
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.periodic_prefix_count._models import (
    MAX_PREFIX_CUTOFF_DIGITS,
    PeriodicUnionPrefixCountResult,
)

__all__ = ["compute_periodic_union_prefix_count"]

MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _reject(
    code: str, message: str, *, location: tuple[str | int, ...] = ("source",)
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"number_theory.periodic_prefix.{code}",
        message=message,
    )


def _count_congruence_prefix(residue: int, modulus: int, upper: int) -> int:
    if upper <= 0:
        return 0
    first = modulus if residue == 0 else residue
    if first > upper:
        return 0
    return 1 + (upper - first) // modulus


def _union_prefix_count(source: PeriodicCongruenceUnionSource, upper: int) -> int:
    """Count one union prefix from compressed generalized-CRT terms."""
    terms: dict[tuple[int, int], int] = {}
    for subset in source.subsets:
        modulus = parse_canonical_integer(subset.modulus)
        for residue_text in subset.residues:
            residue = parse_canonical_integer(residue_text)
            deltas: dict[tuple[int, int], int] = {(residue, modulus): 1}
            for congruence, coefficient in tuple(terms.items()):
                intersection = _merge_congruences(congruence, (residue, modulus))
                if intersection is not None:
                    deltas[intersection] = deltas.get(intersection, 0) - coefficient
            for congruence, delta in deltas.items():
                coefficient = terms.get(congruence, 0) + delta
                if coefficient:
                    terms[congruence] = coefficient
                else:
                    terms.pop(congruence, None)
    count = sum(
        coefficient * _count_congruence_prefix(residue, modulus, upper)
        for (residue, modulus), coefficient in terms.items()
    )
    if not 0 <= count <= upper:
        raise RuntimeError("periodic union prefix count violated its cardinality bound")
    return count


def _prefix_union_count(
    source: PeriodicCongruenceUnionSource,
    plan: _ExecutionPlan,
    upper: int,
) -> int:
    if plan.method == "FULL_UNION":
        return upper
    if plan.method == "PERIOD_LIFT":
        mask = _union_mask(source, plan.common_period)
        count = sum(mask[1 : upper + 1])
        if upper >= plan.common_period and mask[0]:
            count += 1
        return count
    if plan.method == "SPARSE_LIFT":
        occupied = _sparse_union(source, plan.common_period)
        return sum(
            (residue if residue else plan.common_period) <= upper
            for residue in occupied
        )
    return _union_prefix_count(source, upper)


def _admit_source(source: PeriodicCongruenceUnionSource, cutoff: int) -> _ExecutionPlan:
    if not isinstance(source, PeriodicCongruenceUnionSource):
        _reject("invalid_source", "source must be a canonical periodic union")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        _reject("invalid_cutoff", "cutoff must be a strict integer", location=("cutoff",))
    if cutoff < 0:
        _reject("negative_cutoff", "cutoff must be nonnegative", location=("cutoff",))
    if cutoff > 10**MAX_PREFIX_CUTOFF_DIGITS - 1:
        _reject(
            "cutoff_digit_bound",
            f"cutoff must have at most {MAX_PREFIX_CUTOFF_DIGITS} digits",
            location=("cutoff",),
        )
    try:
        plan = require_admitted_periodic_source(source)
    except ValueError as exc:
        _reject("execution_bound", str(exc))
    period = plan.common_period
    cutoff_text = format_canonical_integer(cutoff)
    try:
        source_bytes = len(encode_strict_json(source.model_dump(mode="json")))
    except ValueError as exc:
        _reject("source_representation", str(exc))
    scalar_bytes = (
        source_bytes
        + len(cutoff_text) * 2
        + len(format_canonical_integer(period))
        + 256
    )
    if scalar_bytes > MAX_RESULT_BYTES:
        _reject(
            "result_size_bound",
            f"the scalar result exceeds the {MAX_RESULT_BYTES}-byte output bound",
        )
    return plan


def compute_periodic_union_prefix_count(
    source: PeriodicCongruenceUnionSource,
    cutoff: int,
) -> PeriodicUnionPrefixCountResult:
    """Return the exact count of integers in [1, cutoff] belonging to the periodic set.

    Uses the periodicity: if the common period is L and c residues are
    occupied in one period, then the count through X is q*c + r,
    where q = X // L, r = X % L, and c is the one-period count.
    """
    plan = _admit_source(source, cutoff)
    period = plan.common_period
    occupied_count = _prefix_union_count(source, plan, period)
    partial_union_count = _prefix_union_count(source, plan, cutoff % period)
    if source.complement:
        occupied_count = period - occupied_count
        partial_union_count = cutoff % period - partial_union_count
    q, _remainder = divmod(cutoff, period)
    count = q * occupied_count + partial_union_count
    cutoff_text = format_canonical_integer(cutoff)
    period_text = format_canonical_integer(period)
    occupied_text = format_canonical_integer(occupied_count)
    count_text = format_canonical_integer(count)
    return PeriodicUnionPrefixCountResult(
        source=source,
        cutoff=cutoff_text,
        common_period=period_text,
        occupied_count=occupied_text,
        count=count_text,
    )
