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
    use_inclusion_exclusion: bool
    use_singleton_closed_form: bool


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
    use_inclusion_exclusion = False
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
        use_singleton_closed_form = bool(edge_masks) and all(
            edge_mask & (edge_mask - 1) == 0 for edge_mask in edge_masks
        )
        active_state_count = 1 << len(active_vertices)
        ie_terms = _inclusion_exclusion_terms(len(edge_masks), n)
        direct_work = len(edge_masks) * active_state_count + active_state_count * (
            n - len(active_vertices) + 1
        )
        ie_work = ie_terms * (n + len(edge_masks) + 2) if ie_terms is not None else None
        use_inclusion_exclusion = (
            ie_work is not None and ie_work < direct_work
        ) and not use_singleton_closed_form
        if (
            active_state_count > MAX_SUBSET_STATES
            and not use_inclusion_exclusion
            and not use_singleton_closed_form
        ):
            raise OperationDomainValidationError(
                location=("hypergraph", "vertices"),
                code="hypergraph_containment.state_bound_exceeded",
                message="the active subset-state envelope is exceeded",
            )
        lift_work = active_state_count * (n - len(active_vertices) + 1)
        if (
            not use_inclusion_exclusion
            and not use_singleton_closed_form
            and (
                len(edge_masks) * active_state_count + lift_work > MAX_CONTAINMENT_WORK
            )
        ):
            raise OperationDomainValidationError(
                location=("hypergraph", "edges"),
                code="hypergraph_containment.work_bound_exceeded",
                message="the complete subset containment work envelope is exceeded",
            )
    else:
        edge_masks = ()
        active_vertices = ()
        use_singleton_closed_form = False
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
    return _ContainmentAdmissionPlan(
        edge_masks,
        active_vertices,
        use_inclusion_exclusion,
        use_singleton_closed_form,
    )


def _inclusion_exclusion_terms(edge_count: int, vertex_count: int) -> int | None:
    """Return the charged term count when edge-subset inclusion-exclusion fits."""

    if edge_count >= 63:
        return None
    terms = (1 << edge_count) - 1
    # Each term contributes one union mask, one probability power, and all
    # n+1 profile entries.  Keep this complete regime under the same work
    # envelope as subset enumeration.
    # Each term scans every edge mask to build the union mask, plus one
    # union/probability computation and all n+1 profile entries.
    if terms * (vertex_count + edge_count + 2) > MAX_CONTAINMENT_WORK:
        return None
    return terms


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

    if plan.use_singleton_closed_form:
        # Every minimal edge is a singleton over one distinct vertex, so the
        # event "a declared edge is contained" reduces to "at least one of the
        # m singleton-edge vertices is retained".  This admits a closed form
        # instead of enumerating the (possibly exponential) active states.
        m = len(plan.edge_masks)
        singleton_counts = tuple(comb(n, k) - comb(n - m, k) for k in range(n + 1))
        success_count = (1 << n) - (1 << (n - m))
        p = retention_probability.as_fraction()
        probability = Fraction(1) - (1 - p) ** m
        return HypergraphVertexContainmentResult(
            hypergraph=hypergraph,
            retention_probability=retention_probability,
            containing_subset_counts=tuple(
                format_canonical_integer(value) for value in singleton_counts
            ),
            total_state_count=format_canonical_integer(1 << n),
            success_count=format_canonical_integer(success_count),
            probability=CanonicalRational.from_fraction(probability),
        )

    active_n = len(plan.active_vertices)
    isolated_n = n - active_n
    if plan.use_inclusion_exclusion:
        edge_count = len(plan.edge_masks)
        ie_counts = [0] * (n + 1)
        success_count = 0
        p = retention_probability.as_fraction()
        probability = Fraction(0)
        for subset in range(1, 1 << edge_count):
            union_mask = 0
            selected = 0
            for edge_index, edge_mask in enumerate(plan.edge_masks):
                if subset & (1 << edge_index):
                    union_mask |= edge_mask
                    selected += 1
            union_size = union_mask.bit_count()
            sign = 1 if selected % 2 else -1
            success_count += sign * (1 << (n - union_size))
            probability += sign * p**union_size
            for k in range(union_size, n + 1):
                ie_counts[k] += sign * comb(n - union_size, k - union_size)
        return HypergraphVertexContainmentResult(
            hypergraph=hypergraph,
            retention_probability=retention_probability,
            containing_subset_counts=tuple(
                format_canonical_integer(value) for value in ie_counts
            ),
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
