"""Hypergraph vertex containment probability kernel."""

from __future__ import annotations

from fractions import Fraction
from math import comb

from jacobian._exact import (
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.probability.hypergraph_containment._models import (
    MAX_CONTAINMENT_WORK,
    MAX_SUBSET_STATES,
    HypergraphVertexContainmentResult,
)

__all__ = ["compute_hypergraph_vertex_containment"]

MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _admit_hypergraph_vertex_containment(
    hypergraph: FiniteHypergraph,
    retention_probability: CanonicalRational,
) -> tuple[int, ...]:
    if not isinstance(hypergraph, FiniteHypergraph):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph_containment.invalid_hypergraph",
            message="hypergraph must be a FiniteHypergraph value",
        )
    if not isinstance(retention_probability, CanonicalRational):
        raise OperationDomainValidationError(
            location=("retention_probability",),
            code="hypergraph_containment.invalid_probability",
            message="retention_probability must be a CanonicalRational value",
        )
    p = retention_probability.as_fraction()
    if not 0 <= p <= 1:
        raise OperationDomainValidationError(
            location=("retention_probability",),
            code="hypergraph_containment.probability_out_of_range",
            message="retention_probability must be between 0 and 1",
        )
    trivial_event = not hypergraph.edges or any(
        not members for _, members in hypergraph.edges
    )
    n = len(hypergraph.vertices)
    state_count = 1 << n
    if not trivial_event:
        if state_count > MAX_SUBSET_STATES:
            raise OperationDomainValidationError(
                location=("hypergraph", "vertices"),
                code="hypergraph_containment.state_bound_exceeded",
                message="the complete subset-state envelope is exceeded",
            )
        vertex_index = {vertex: index for index, vertex in enumerate(hypergraph.vertices)}
        edge_masks = tuple(
            dict.fromkeys(
                sum(1 << vertex_index[member] for member in members)
                for _, members in hypergraph.edges
            )
        )
        if len(edge_masks) * state_count > MAX_CONTAINMENT_WORK:
            raise OperationDomainValidationError(
                location=("hypergraph", "edges"),
                code="hypergraph_containment.work_bound_exceeded",
                message="the complete subset containment work envelope is exceeded",
            )
    else:
        edge_masks = ()
    support_size = len(
        {member for _, members in hypergraph.edges for member in members}
    )
    probability_digits = (
        1
        if trivial_event
        else support_size * canonical_rational_component_digits(retention_probability)
    )
    if probability_digits > 32_768:
        raise OperationDomainValidationError(
            location=("retention_probability",),
            code="hypergraph_containment.result_growth_exceeded",
            message="probability rational growth exceeds the canonical digit envelope",
        )
    try:
        result_probe = {
            "hypergraph": hypergraph.model_dump(mode="json"),
            "retention_probability": retention_probability.model_dump(mode="json"),
            "containing_subset_counts": [
                10 ** len(str(comb(n, k))) - 1 for k in range(n + 1)
            ],
            "total_state_count": state_count,
            "success_count": state_count,
            "probability": {
                "num": "9" * max(1, probability_digits),
                "den": "9" * max(1, probability_digits),
            },
        }
        result_bytes = len(encode_strict_json(result_probe))
    except CanonicalizationError as exc:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph_containment.result_size_bound",
            message="the complete containment profile exceeds the canonical output bound",
        ) from exc
    if result_bytes > MAX_RESULT_BYTES:
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="hypergraph_containment.result_size_bound",
            message="the complete containment profile exceeds the canonical output bound",
        )
    return edge_masks


def compute_hypergraph_vertex_containment(
    hypergraph: FiniteHypergraph,
    retention_probability: CanonicalRational,
) -> HypergraphVertexContainmentResult:
    """Return the complete vertex-containment profile of a hypergraph.

    For each k-subset of vertices, check if it contains a declared hyperedge.
    Count by k and compute the exact probability under independent vertex
    retention.
    """
    edge_masks = _admit_hypergraph_vertex_containment(hypergraph, retention_probability)
    n = len(hypergraph.vertices)

    if not hypergraph.edges:
        return HypergraphVertexContainmentResult(
            hypergraph=hypergraph,
            retention_probability=retention_probability,
            containing_subset_counts=tuple(0 for _ in range(n + 1)),
            total_state_count=1 << n,
            success_count=0,
            probability=CanonicalRational.from_fraction(Fraction(0)),
        )
    if any(not members for _, members in hypergraph.edges):
        all_counts = tuple(comb(n, k) for k in range(n + 1))
        return HypergraphVertexContainmentResult(
            hypergraph=hypergraph,
            retention_probability=retention_probability,
            containing_subset_counts=all_counts,
            total_state_count=1 << n,
            success_count=1 << n,
            probability=CanonicalRational.from_fraction(Fraction(1)),
        )

    counts: list[int] = [0] * (n + 1)
    for mask in range(1 << n):
        k = mask.bit_count()
        contains_edge = any(edge_mask & ~mask == 0 for edge_mask in edge_masks)
        if contains_edge:
            counts[k] += 1

    total = 1 << n
    success = sum(counts)
    p = retention_probability.as_fraction()
    q = 1 - p
    prob = Fraction(0)
    for k in range(n + 1):
        prob += counts[k] * (p**k) * (q ** (n - k))

    return HypergraphVertexContainmentResult(
        hypergraph=hypergraph,
        retention_probability=retention_probability,
        containing_subset_counts=tuple(counts),
        total_state_count=total,
        success_count=success,
        probability=CanonicalRational.from_fraction(prob),
    )
