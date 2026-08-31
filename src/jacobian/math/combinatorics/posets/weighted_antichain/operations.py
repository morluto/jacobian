"""Maximum weight antichain kernel using min-cut reduction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm as _lcm

import networkx as nx

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.combinatorics.posets.weighted_antichain._models import (
    MAX_ENUMERATION_WORK,
    MaximumWeightAntichainResult,
)

__all__ = ["compute_maximum_weight_antichain"]


@dataclass(frozen=True, slots=True)
class _MaximumWeightAntichainAdmission:
    weights: tuple[Fraction, ...]


def _admit_maximum_weight_antichain(
    poset: FinitePoset, weights: tuple[CanonicalRational, ...]
) -> _MaximumWeightAntichainAdmission:
    if not isinstance(poset, FinitePoset):
        raise OperationDomainValidationError(
            location=("poset",),
            code="weighted_antichain.invalid_poset",
            message="poset must be a FinitePoset value",
        )
    if not isinstance(weights, tuple) or len(weights) != len(poset.elements):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.weight_count_mismatch",
            message="weights must have exactly one entry per poset element",
        )
    if any(not isinstance(weight, CanonicalRational) for weight in weights):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.invalid_weight",
            message="weights must be CanonicalRational values",
        )
    weight_fracs = tuple(weight.as_fraction() for weight in weights)
    if any(weight < 0 for weight in weight_fracs):
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.negative_weight",
            message="all weights must be nonnegative",
        )
    n = len(poset.elements)
    if n > MAX_ENUMERATION_WORK:
        raise OperationDomainValidationError(
            location=("poset", "elements"),
            code="weighted_antichain.work_bound_exceeded",
            message="the antichain work envelope is exceeded",
        )
    width = _poset_width(poset, list(poset.elements))
    max_digits = max(
        (canonical_rational_component_digits(weight) for weight in weights),
        default=1,
    )
    arithmetic_work = max(width, 1) * max_digits
    if arithmetic_work > MAX_ENUMERATION_WORK:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.arithmetic_work_bound_exceeded",
            message="rational antichain summation exceeds the admitted work envelope",
        )
    # Bound the common-denominator expansion: the LCM of all denominators
    # can have up to width * max_digits digits (in the worst case), and each
    # of n scaled weights then has similar size.  Charge this before execution.
    denominators = [w.denominator for w in weight_fracs]
    unique_denoms = len(set(denominators))
    if unique_denoms <= 1:
        max_sum_digits = max_digits + (len(str(width)) if width > 1 else 0)
    else:
        # The sum of up to width rationals with different denominators has
        # at most width * max_digits digits in the numerator and the LCM of
        # denominators (at most width * max_digits digits) in the denominator.
        max_sum_digits = max(width, 1) * max_digits + len(str(max(width, 1)))
    if max_sum_digits > 32_768:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.result_growth_exceeded",
            message="maximum-weight rational growth exceeds the canonical digit envelope",
        )
    rational_size = strict_json_object_size(
        (
            ("num", len(encode_strict_json("9" * max_sum_digits))),
            ("den", len(encode_strict_json("9" * max_sum_digits))),
        )
    )
    labels_size = (
        2
        + max(n - 1, 0)
        + sum(len(encode_strict_json(element)) for element in poset.elements)
    )
    result_bytes = strict_json_object_size(
        (
            ("poset", len(encode_strict_json(poset.model_dump(mode="json")))),
            (
                "weights",
                len(
                    encode_strict_json(
                        [weight.model_dump(mode="json") for weight in weights]
                    )
                ),
            ),
            ("maximum_weight", rational_size),
            ("antichain", labels_size),
        )
    )
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("poset", "weights"),
            code="weighted_antichain.result_too_large",
            message="maximum-weight antichain result exceeds the canonical output envelope",
        )
    return _MaximumWeightAntichainAdmission(weights=weight_fracs)


def compute_maximum_weight_antichain(
    poset: FinitePoset,
    weights: tuple[CanonicalRational, ...],
) -> MaximumWeightAntichainResult:
    """Return the exact maximum weight antichain and a deterministic witness.

    Uses a polynomial min-cut reduction via the bipartite vertex cover
    formulation: the maximum weight antichain on a poset equals the total
    weight minus the minimum weight vertex cover of the bipartite graph
    induced by the order relation.
    """
    admission = _admit_maximum_weight_antichain(poset, weights)
    elements = list(poset.elements)
    n = len(elements)
    weight_fracs = admission.weights

    if n == 0:
        return MaximumWeightAntichainResult(
            poset=poset,
            weights=weights,
            maximum_weight=CanonicalRational.from_fraction(Fraction(0)),
            antichain=(),
        )

    idx = {e: i for i, e in enumerate(elements)}
    order_pairs = [
        (idx[pair.lower], idx[pair.upper]) for pair in poset.strict_order_pairs
    ]

    # Convert fractions to common denominator for integer flow network.
    common_den = 1
    for w in weight_fracs:
        common_den = _lcm(common_den, w.denominator)
    int_weights = [int(w * common_den) for w in weight_fracs]
    total_weight_int = sum(int_weights)

    # Build NetworkX flow network
    source, sink = "source", "sink"
    g = nx.DiGraph()
    g.add_edge(source, sink, capacity=0)  # ensure nodes exist

    for i in range(n):
        w = int_weights[i]
        g.add_edge(source, f"L{i}", capacity=w)
        g.add_edge(f"R{i}", sink, capacity=w)

    # Use total_weight_int + 1 as infinite capacity (always > any cut)
    inf_cap = total_weight_int + 1
    for v, u in order_pairs:
        g.add_edge(f"L{v}", f"R{u}", capacity=inf_cap)

    min_cut_value, partition = nx.minimum_cut(g, source, sink)
    max_antichain_weight_int = total_weight_int - min_cut_value

    # partition[0] = source side, partition[1] = sink side
    reachable = partition[0]
    in_vertex_cover = set()
    for i in range(n):
        if f"L{i}" not in reachable:
            in_vertex_cover.add(i)
        if f"R{i}" in reachable:
            in_vertex_cover.add(i)

    antichain_indices = [i for i in range(n) if i not in in_vertex_cover]
    antichain_indices.sort()

    best_weight = Fraction(max_antichain_weight_int, common_den)
    best_antichain = tuple(elements[i] for i in antichain_indices)

    return MaximumWeightAntichainResult(
        poset=poset,
        weights=weights,
        maximum_weight=CanonicalRational.from_fraction(best_weight),
        antichain=best_antichain,
    )


def _poset_width(poset: FinitePoset, elements: list[str]) -> int:
    """Return the maximum antichain size via the bipartite matching theorem."""
    index = {element: position for position, element in enumerate(elements)}
    adjacent: list[list[int]] = [[] for _ in elements]
    for pair in poset.strict_order_pairs:
        adjacent[index[pair.lower]].append(index[pair.upper])

    matched_upper = [-1] * len(elements)

    def augment(lower: int, seen: set[int]) -> bool:
        for upper in adjacent[lower]:
            if upper in seen:
                continue
            seen.add(upper)
            if matched_upper[upper] == -1 or augment(matched_upper[upper], seen):
                matched_upper[upper] = lower
                return True
        return False

    matching = sum(augment(lower, set()) for lower in range(len(elements)))
    return len(elements) - matching
