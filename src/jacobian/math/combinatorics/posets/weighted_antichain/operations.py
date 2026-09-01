"""Maximum weight antichain kernel using min-cut reduction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from math import lcm as _lcm
from typing import Any

import networkx as nx

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
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
        extra = math.ceil(math.log10(width)) if width > 1 else 0
        max_sum_digits = max_digits + extra
    else:
        # The sum of up to width rationals with different denominators has
        # at most width * max_digits digits in the numerator and the LCM of
        # denominators (at most width * max_digits digits) in the denominator.
        extra = math.ceil(math.log10(width)) if width > 1 else 0
        max_sum_digits = max(width, 1) * max_digits + extra
    if max_sum_digits > 32_768:
        raise OperationDomainValidationError(
            location=("weights",),
            code="weighted_antichain.result_growth_exceeded",
            message="maximum-weight rational growth exceeds the canonical digit envelope",
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
    common_den: int = 1
    for w in weight_fracs:
        common_den = _lcm(common_den, w.denominator)
    int_weights = [int(w * common_den) for w in weight_fracs]
    total_weight_int: int = sum(int_weights)

    # Build NetworkX flow network
    source, sink = "source", "sink"
    g: Any = nx.DiGraph()
    g.add_edge(source, sink, capacity=0)  # ensure nodes exist

    for i in range(n):
        weight_int = int_weights[i]
        g.add_edge(source, f"L{i}", capacity=weight_int)
        g.add_edge(f"R{i}", sink, capacity=weight_int)

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

    best_weight = Fraction(int(max_antichain_weight_int), int(common_den))
    best_antichain = tuple(elements[i] for i in antichain_indices)

    return MaximumWeightAntichainResult(
        poset=poset,
        weights=weights,
        maximum_weight=CanonicalRational.from_fraction(best_weight),
        antichain=best_antichain,
    )


def _poset_width(poset: FinitePoset, elements: list[str]) -> int:
    """Return the maximum antichain size via the bipartite matching theorem."""
    if not elements:
        return 0
    # Delegate width matching to the maintained NetworkX bipartite primitive
    # (core/operations.py:171-182), avoiding a second correctness-sensitive
    # augmenting-path kernel in this module.
    left_nodes = [(0, element) for element in elements]
    right_nodes = [(1, element) for element in elements]
    graph: Any = nx.Graph()
    graph.add_nodes_from(left_nodes, bipartite=0)
    graph.add_nodes_from(right_nodes, bipartite=1)
    graph.add_edges_from(
        ((0, pair.lower), (1, pair.upper)) for pair in poset.strict_order_pairs
    )
    raw_matching = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=left_nodes)
    matching_size = len(raw_matching) // 2
    return len(elements) - matching_size
