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
    _intersection_bounds,
    _merge_congruences,
    common_period,
    require_admitted_periodic_source,
)
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.periodic_prefix_count._models import (
    PeriodicUnionPrefixCountResult,
)

__all__ = ["compute_periodic_union_prefix_count"]

MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("source",),
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


def _admit_source(source: PeriodicCongruenceUnionSource, cutoff: int) -> int:
    if not isinstance(source, PeriodicCongruenceUnionSource):
        _reject("invalid_source", "source must be a canonical periodic union")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        _reject("invalid_cutoff", "cutoff must be a strict integer")
    if cutoff < 0:
        _reject("negative_cutoff", "cutoff must be nonnegative")
    try:
        require_admitted_periodic_source(source)
        period = common_period(source)
    except ValueError as exc:
        _reject("execution_bound", str(exc))
    states, merges = _intersection_bounds(source)
    full_union = any(
        len(subset.residues) == parse_canonical_integer(subset.modulus)
        for subset in source.subsets
    )
    if not full_union and (states > 65_535 or merges > 100_000):
        _reject(
            "prefix_execution_bound",
            "scalar prefix counting requires a bounded compressed congruence profile",
        )
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
    return period


def compute_periodic_union_prefix_count(
    source: PeriodicCongruenceUnionSource,
    cutoff: int,
) -> PeriodicUnionPrefixCountResult:
    """Return the exact count of integers in [1, cutoff] belonging to the periodic set.

    Uses the periodicity: if the common period is L and c residues are
    occupied in one period, then the count through X is q*c + r,
    where q = X // L, r = X % L, and c is the one-period count.
    """
    period = _admit_source(source, cutoff)
    full_union = any(
        len(subset.residues) == parse_canonical_integer(subset.modulus)
        for subset in source.subsets
    )
    if full_union:
        occupied_count = period
        partial_union_count = cutoff % period
    else:
        occupied_count = _union_prefix_count(source, period)
        partial_union_count = _union_prefix_count(source, cutoff % period)
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
