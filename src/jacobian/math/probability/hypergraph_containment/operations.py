"""Hypergraph vertex containment probability kernel."""

from __future__ import annotations

from dataclasses import dataclass
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
    format_canonical_integer,
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


@dataclass(frozen=True, slots=True)
class _ContainmentAdmissionPlan:
    edge_masks: tuple[int, ...]
    active_vertices: tuple[int, ...]


def _admit_hypergraph_vertex_containment(
    hypergraph: FiniteHypergraph,
    retention_probability: CanonicalRational,
) -> _ContainmentAdmissionPlan:
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
        vertex_index = {
            vertex: index for index, vertex in enumerate(hypergraph.vertices)
        }
        unique_masks = tuple(
            dict.fromkeys(
                sum(1 << vertex_index[member] for member in members)
                for _, members in hypergraph.edges
            )
        )
        minimal_masks = _minimal_edge_masks(unique_masks)
        support_mask = 0
        for mask in minimal_masks:
            support_mask |= mask
        active_vertices = tuple(
            index for index in range(n) if support_mask & (1 << index)
        )
        active_position = {
            index: position for position, index in enumerate(active_vertices)
        }
        edge_masks = tuple(
            sum(
                1 << active_position[index] for index in range(n) if mask & (1 << index)
            )
            for mask in minimal_masks
        )
        active_state_count = 1 << len(active_vertices)
        small_edge_family = len(edge_masks) <= 2
        if active_state_count > MAX_SUBSET_STATES and not small_edge_family:
            raise OperationDomainValidationError(
                location=("hypergraph", "vertices"),
                code="hypergraph_containment.state_bound_exceeded",
                message="the active subset-state envelope is exceeded",
            )
        lift_work = active_state_count * (n - len(active_vertices) + 1)
        if not small_edge_family and (
            len(edge_masks) * active_state_count + lift_work > MAX_CONTAINMENT_WORK
        ):
            raise OperationDomainValidationError(
                location=("hypergraph", "edges"),
                code="hypergraph_containment.work_bound_exceeded",
                message="the complete subset containment work envelope is exceeded",
            )
    else:
        edge_masks = ()
        active_vertices = ()
    support_size = len(active_vertices)
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
                format_canonical_integer(10 ** len(str(comb(n, k))) - 1)
                for k in range(n + 1)
            ],
            "total_state_count": format_canonical_integer(state_count),
            "success_count": format_canonical_integer(state_count),
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
    return _ContainmentAdmissionPlan(edge_masks, active_vertices)


def _minimal_edge_masks(unique_masks: tuple[int, ...]) -> list[int]:
    """Reduce edge masks to an inclusion antichain with charged comparisons."""

    minimal_masks: list[int] = []
    comparisons = 0
    for mask in sorted(unique_masks, key=int.bit_count):
        dominated = False
        for existing in minimal_masks:
            comparisons += 1
            if comparisons > MAX_CONTAINMENT_WORK:
                raise OperationDomainValidationError(
                    location=("hypergraph", "edges"),
                    code="hypergraph_containment.work_bound_exceeded",
                    message="the antichain reduction work envelope is exceeded",
                )
            if existing & mask == existing:
                dominated = True
                break
        if not dominated:
            minimal_masks.append(mask)
    return minimal_masks


def compute_hypergraph_vertex_containment(
    hypergraph: FiniteHypergraph,
    retention_probability: CanonicalRational,
) -> HypergraphVertexContainmentResult:
    """Return the complete vertex-containment profile of a hypergraph.

    For each k-subset of vertices, check if it contains a declared hyperedge.
    Count by k and compute the exact probability under independent vertex
    retention.
    """
    plan = _admit_hypergraph_vertex_containment(hypergraph, retention_probability)
    edge_masks = plan.edge_masks
    n = len(hypergraph.vertices)

    if not hypergraph.edges:
        return HypergraphVertexContainmentResult(
            hypergraph=hypergraph,
            retention_probability=retention_probability,
            containing_subset_counts=tuple("0" for _ in range(n + 1)),
            total_state_count=format_canonical_integer(1 << n),
            success_count="0",
            probability=CanonicalRational.from_fraction(Fraction(0)),
        )
    if any(not members for _, members in hypergraph.edges):
        all_counts = tuple(comb(n, k) for k in range(n + 1))
        return HypergraphVertexContainmentResult(
            hypergraph=hypergraph,
            retention_probability=retention_probability,
            containing_subset_counts=tuple(
                format_canonical_integer(value) for value in all_counts
            ),
            total_state_count=format_canonical_integer(1 << n),
            success_count=format_canonical_integer(1 << n),
            probability=CanonicalRational.from_fraction(Fraction(1)),
        )

    active_n = len(plan.active_vertices)
    isolated_n = n - active_n
    if len(plan.edge_masks) <= 2:
        edge_sizes = tuple(mask.bit_count() for mask in plan.edge_masks)
        union_size = (
            (plan.edge_masks[0] | plan.edge_masks[1]).bit_count()
            if len(plan.edge_masks) == 2
            else edge_sizes[0]
        )
        single_edge_counts = tuple(
            format_canonical_integer(
                sum(
                    comb(n - edge_size, k - edge_size)
                    for edge_size in edge_sizes
                    if k >= edge_size
                )
                - (
                    comb(n - union_size, k - union_size)
                    if len(edge_sizes) == 2 and k >= union_size
                    else 0
                )
            )
            for k in range(n + 1)
        )
        p = retention_probability.as_fraction()
        success_count = sum(
            1 << (n - edge_size) for edge_size in edge_sizes
        ) - (
            (1 << (n - union_size)) if len(edge_sizes) == 2 else 0
        )
        probability = sum(
            (p**edge_size for edge_size in edge_sizes), Fraction(0)
        ) - (
            p**union_size if len(edge_sizes) == 2 else 0
        )
        return HypergraphVertexContainmentResult(
            hypergraph=hypergraph,
            retention_probability=retention_probability,
            containing_subset_counts=single_edge_counts,
            total_state_count=format_canonical_integer(1 << n),
            success_count=format_canonical_integer(success_count),
            probability=CanonicalRational.from_fraction(probability),
        )

    active_counts: list[int] = [0] * (active_n + 1)
    counts: list[int] = [0] * (n + 1)
    for mask in range(1 << active_n):
        k = mask.bit_count()
        contains_edge = any(edge_mask & ~mask == 0 for edge_mask in edge_masks)
        if contains_edge:
            active_counts[k] += 1
            for isolated_k in range(isolated_n + 1):
                counts[k + isolated_k] += comb(isolated_n, isolated_k)

    total = 1 << n
    success = sum(counts)
    p = retention_probability.as_fraction()
    q = 1 - p
    prob = Fraction(0)
    for k in range(active_n + 1):
        prob += active_counts[k] * (p**k) * (q ** (active_n - k))

    return HypergraphVertexContainmentResult(
        hypergraph=hypergraph,
        retention_probability=retention_probability,
        containing_subset_counts=tuple(
            format_canonical_integer(value) for value in counts
        ),
        total_state_count=format_canonical_integer(total),
        success_count=format_canonical_integer(success),
        probability=CanonicalRational.from_fraction(prob),
    )
