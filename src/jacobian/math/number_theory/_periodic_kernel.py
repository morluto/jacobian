"""Bounded exact kernels for finite congruence unions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.number_theory._periodic_models import (
    MAX_INTERSECTION_MERGES,
    MAX_INTERSECTION_STATES,
    MAX_MATERIALIZED_RESIDUES,
    MAX_PERIOD_LIFT_WORK,
    MAX_PERIOD_SCAN,
    MAX_PERIODIC_INTEGER_DIGITS,
    MAX_PERIODIC_RESULT_BYTES,
    MAX_SPARSE_LIFTED_ROWS,
    PERIODIC_PROFILE_RESULT_ENVELOPE_BYTES,
    PeriodicCongruenceUnionSource,
)

# Operations perform one kernel pass and their authoritative result validators
# replay that operation's defining invariant once. These are per-pass caps, so
# a full accepted call performs at most twice the admitted scan/lift or merge work.


@dataclass(frozen=True, slots=True)
class _ExecutionPlan:
    method: Literal["FULL_UNION", "PERIOD_LIFT", "SPARSE_LIFT", "INCLUSION_EXCLUSION"]
    common_period: int
    lift_work: int


def common_period(source: PeriodicCongruenceUnionSource) -> int:
    """Return the lcm of every normalized source modulus, or one if empty."""

    period = 1
    for subset in source.subsets:
        period = math.lcm(period, parse_canonical_integer(subset.modulus))
        if len(format_canonical_integer(period)) > MAX_PERIODIC_INTEGER_DIGITS:
            raise ValueError("common period exceeds the 256-digit exact-result bound")
    return period


def _lift_work(source: PeriodicCongruenceUnionSource, period: int) -> int:
    return sum(
        len(subset.residues) * (period // parse_canonical_integer(subset.modulus))
        for subset in source.subsets
    )


def _contains_full_subset(source: PeriodicCongruenceUnionSource) -> bool:
    return any(
        len(subset.residues) == parse_canonical_integer(subset.modulus)
        for subset in source.subsets
    )


def _intersection_bounds(
    source: PeriodicCongruenceUnionSource,
) -> tuple[int, int]:
    """Bound retained intersection states and attempted two-class merges.

    Within one modulus at most one distinct residue can occur in a consistent
    intersection.  If ``s`` states precede a modulus with ``r`` residues, the
    retained-state bound after that modulus is ``(s + 1)(r + 1) - 1``.  The
    merge bound also counts the inconsistent attempts against residues already
    processed at that same modulus.
    """

    states = 0
    merges = 0
    for subset in source.subsets:
        previous_states = states
        for residue_index in range(len(subset.residues)):
            states_before_residue = (previous_states + 1) * (residue_index + 1) - 1
            merges += states_before_residue
            if merges > MAX_INTERSECTION_MERGES:
                return states_before_residue, merges
        states = (previous_states + 1) * (len(subset.residues) + 1) - 1
        if states > MAX_INTERSECTION_STATES:
            return states, merges
    return states, merges


def require_admitted_periodic_source(
    source: PeriodicCongruenceUnionSource,
) -> _ExecutionPlan:
    """Select a complete exact regime before expanding or intersecting."""

    period = common_period(source)
    if _contains_full_subset(source):
        return _ExecutionPlan(
            method="FULL_UNION",
            common_period=period,
            lift_work=0,
        )
    lift_work = _lift_work(source, period)
    if period <= MAX_PERIOD_SCAN and period + lift_work <= MAX_PERIOD_LIFT_WORK:
        return _ExecutionPlan(
            method="PERIOD_LIFT",
            common_period=period,
            lift_work=lift_work,
        )
    if lift_work <= MAX_SPARSE_LIFTED_ROWS:
        return _ExecutionPlan(
            method="SPARSE_LIFT",
            common_period=period,
            lift_work=lift_work,
        )
    states, merges = _intersection_bounds(source)
    if states <= MAX_INTERSECTION_STATES and merges <= MAX_INTERSECTION_MERGES:
        return _ExecutionPlan(
            method="INCLUSION_EXCLUSION",
            common_period=period,
            lift_work=lift_work,
        )
    raise ValueError(
        "periodic union exceeds all exact execution regimes: one-period lift "
        f"requires period+lift work {period + lift_work} "
        f"(limit {MAX_PERIOD_LIFT_WORK}), sparse lifting requires {lift_work} "
        f"lifted rows/states (limit {MAX_SPARSE_LIFTED_ROWS}), while compressed "
        "inclusion-exclusion "
        f"requires at most {states} states/{merges} merges "
        f"(limits {MAX_INTERSECTION_STATES}/{MAX_INTERSECTION_MERGES})"
    )


def require_materializable_periodic_source(
    source: PeriodicCongruenceUnionSource,
) -> _ExecutionPlan:
    """Require the complete residue rows and their construction to fit."""

    plan = require_admitted_periodic_source(source)
    if plan.method == "FULL_UNION":
        if not source.complement and plan.common_period > MAX_MATERIALIZED_RESIDUES:
            raise ValueError(
                "materialized full union exceeds the conservative "
                f"{MAX_MATERIALIZED_RESIDUES}-residue output bound"
            )
        result_rows = 0 if source.complement else plan.common_period
    else:
        if source.complement:
            if plan.common_period > MAX_MATERIALIZED_RESIDUES:
                raise ValueError(
                    "complemented profile common period exceeds the conservative "
                    f"{MAX_MATERIALIZED_RESIDUES}-residue output bound"
                )
            materialization_work = plan.common_period + plan.lift_work
            result_rows = plan.common_period
        else:
            if plan.lift_work > MAX_MATERIALIZED_RESIDUES:
                raise ValueError(
                    "materialized union exceeds the conservative "
                    f"{MAX_MATERIALIZED_RESIDUES}-residue output bound"
                )
            materialization_work = plan.lift_work
            result_rows = min(plan.common_period, plan.lift_work)
        if materialization_work > MAX_PERIOD_LIFT_WORK:
            raise ValueError(
                "materialized profile exceeds the bounded one-period lift work"
            )
    period_digits = len(format_canonical_integer(plan.common_period - 1))
    source_bytes = len(encode_strict_json(source.model_dump(mode="json")))
    estimated_result_bytes = (
        source_bytes
        + result_rows * (period_digits + 3)
        + PERIODIC_PROFILE_RESULT_ENVELOPE_BYTES
    )
    if estimated_result_bytes > MAX_PERIODIC_RESULT_BYTES:
        raise ValueError(
            "materialized profile would exceed the canonical output budget of "
            f"{MAX_PERIODIC_RESULT_BYTES} bytes"
        )
    return plan


def _union_mask(source: PeriodicCongruenceUnionSource, period: int) -> bytearray:
    mask = bytearray(period)
    for subset in source.subsets:
        modulus = parse_canonical_integer(subset.modulus)
        for residue_text in subset.residues:
            residue = parse_canonical_integer(residue_text)
            for value in range(residue, period, modulus):
                mask[value] = 1
    return mask


def _sparse_union(source: PeriodicCongruenceUnionSource, period: int) -> set[int]:
    """Lift the declared classes into one exact set of period representatives."""

    occupied: set[int] = set()
    for subset in source.subsets:
        modulus = parse_canonical_integer(subset.modulus)
        for residue_text in subset.residues:
            residue = parse_canonical_integer(residue_text)
            occupied.update(range(residue, period, modulus))
    return occupied


def _merge_congruences(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[int, int] | None:
    """Merge two congruences through SymPy's maintained generalized CRT."""

    from sympy.ntheory.modular import solve_congruence

    result = solve_congruence(left, right, check=False)
    compatible = (left[0] - right[0]) % math.gcd(left[1], right[1]) == 0
    if result is None:
        if compatible:
            raise RuntimeError("generalized CRT omitted a compatible intersection")
        return None
    residue, modulus = result
    merged = int(residue), int(modulus)
    expected_modulus = math.lcm(left[1], right[1])
    if (
        not compatible
        or merged[1] != expected_modulus
        or not 0 <= merged[0] < merged[1]
        or merged[0] % left[1] != left[0]
        or merged[0] % right[1] != right[0]
    ):
        raise RuntimeError("generalized CRT result violated its defining invariant")
    return merged


def _measure_by_inclusion_exclusion(
    source: PeriodicCongruenceUnionSource, period: int
) -> int:
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
        coefficient * (period // modulus)
        for (_residue, modulus), coefficient in terms.items()
    )
    if not 0 <= count <= period:
        raise RuntimeError("generalized-CRT union count violated its cardinality bound")
    return count


def measure_periodic_union(source: PeriodicCongruenceUnionSource) -> int:
    """Return the exact occupied count, respecting the complement flag."""

    plan = require_admitted_periodic_source(source)
    if plan.method == "FULL_UNION":
        union_count = plan.common_period
    elif plan.method == "PERIOD_LIFT":
        union_count = sum(_union_mask(source, plan.common_period))
    elif plan.method == "SPARSE_LIFT":
        union_count = len(_sparse_union(source, plan.common_period))
    else:
        union_count = _measure_by_inclusion_exclusion(source, plan.common_period)
    return plan.common_period - union_count if source.complement else union_count


def materialize_periodic_union(
    source: PeriodicCongruenceUnionSource,
) -> tuple[int, ...]:
    """Return every occupied representative in canonical increasing order."""

    plan = require_materializable_periodic_source(source)
    if plan.method == "FULL_UNION":
        return () if source.complement else tuple(range(plan.common_period))
    occupied = _sparse_union(source, plan.common_period)
    if source.complement:
        return tuple(
            residue for residue in range(plan.common_period) if residue not in occupied
        )
    return tuple(sorted(occupied))


__all__ = [
    "MAX_INTERSECTION_MERGES",
    "MAX_INTERSECTION_STATES",
    "MAX_PERIODIC_RESULT_BYTES",
    "MAX_PERIOD_LIFT_WORK",
    "MAX_PERIOD_SCAN",
    "MAX_SPARSE_LIFTED_ROWS",
    "common_period",
    "materialize_periodic_union",
    "measure_periodic_union",
    "require_admitted_periodic_source",
    "require_materializable_periodic_source",
]
