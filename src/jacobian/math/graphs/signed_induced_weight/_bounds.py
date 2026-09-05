"""Admission planning for signed induced-weight extrema."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.optimization._models import RationalWeightedGraph

# Total vertices/edges admitted equal the shared weighted-graph carrier. The
# exhaustive search itself is bounded per support component below, so sparse
# graphs with many vertices but small nonzero-weight components are admitted
# while dense graphs beyond any component envelope are still rejected.
MAX_SIGNED_WEIGHT_VERTICES = 32
MAX_SIGNED_WEIGHT_EDGES = 496
# One support component of 20 vertices is the previously admitted exhaustive
# envelope; every component search stays within it.
MAX_SIGNED_WEIGHT_COMPONENT_VERTICES = 20
MAX_SUBSET_ENUMERATION = 1 << MAX_SIGNED_WEIGHT_COMPONENT_VERTICES
MAX_SIGNED_WEIGHT_WORK_UNITS = 25_000_000


@dataclass(frozen=True, slots=True)
class SignedWeightComponent:
    """One support-connected vertex set with its local exhaustive-search charge."""

    vertices: tuple[int, ...]
    adjacency: tuple[tuple[tuple[int, int], ...], ...]
    candidate_subsets: int
    edge_updates: int


@dataclass(frozen=True, slots=True)
class SignedInducedWeightAdmission:
    """One exact integer-scaled component-bounded search plan."""

    denominator: int
    components: tuple[SignedWeightComponent, ...]
    isolates: tuple[int, ...]
    integer_limbs: int
    work_units: int


def _decimal_digit_upper_bound(value: int) -> int:
    """Return a cheap conservative decimal-width bound for an integer."""

    if value == 0:
        return 1
    return (abs(value).bit_length() * 30_103) // 100_000 + 1


def _support_groups(
    order: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    """Group vertex indices joined by support edges via union-find."""

    parent = list(range(order))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in edges:
        root_a, root_b = find(left), find(right)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    groups: dict[int, list[int]] = {}
    for index in range(order):
        groups.setdefault(find(index), []).append(index)
    ordered = sorted((min(group), tuple(group)) for group in groups.values() if group)
    return tuple(group for _, group in ordered)


def _charge_component(
    group: tuple[int, ...],
    edges: tuple[tuple[int, int, int], ...],
    integer_limbs: int,
) -> SignedWeightComponent:
    """Build one component plan with its local Gray-code work charge."""

    position = {global_index: local for local, global_index in enumerate(group)}
    adjacency: list[list[tuple[int, int]]] = [[] for _ in group]
    for left, right, weight in edges:
        if left in position and right in position:
            adjacency[position[left]].append((position[right], weight))
            adjacency[position[right]].append((position[left], weight))
    size = len(group)
    candidate_subsets = 1 << size
    degrees = tuple(len(neighbors) for neighbors in adjacency)
    edge_updates = sum(
        degree * (1 << (size - local - 1)) for local, degree in enumerate(degrees)
    )
    return SignedWeightComponent(
        vertices=group,
        adjacency=tuple(tuple(sorted(neighbors)) for neighbors in adjacency),
        candidate_subsets=candidate_subsets,
        edge_updates=edge_updates,
    )


def _component_work_unit(component: SignedWeightComponent, integer_limbs: int) -> int:
    return component.candidate_subsets + component.edge_updates * integer_limbs


def admit_signed_induced_weight(
    graph: RationalWeightedGraph,
) -> SignedInducedWeightAdmission:
    """Derive one bounded component-search plan shared by native and wire calls.

    Vertices joined by nonzero-weight edges form support components; induced
    weights add across components, so each component is optimized by exact
    Gray-code search and the totals charged together. Zero-weight edges stay
    in the retained source but never enter the search support: their
    endpoints are isolates for witness purposes.
    """

    if not isinstance(graph, RationalWeightedGraph):
        raise TypeError("signed_induced_weight_extrema expects a RationalWeightedGraph")
    order = len(graph.vertices)
    if order > MAX_SIGNED_WEIGHT_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.signed_induced_weight.vertex_bound",
            message=(
                "signed induced-weight extrema support at most "
                f"{MAX_SIGNED_WEIGHT_VERTICES} vertices"
            ),
        )
    if len(graph.edges) > MAX_SIGNED_WEIGHT_EDGES:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_induced_weight.edge_bound",
            message=(
                "signed induced-weight extrema support at most "
                f"{MAX_SIGNED_WEIGHT_EDGES} edges"
            ),
        )

    fractions = tuple(edge.weight.as_fraction() for edge in graph.edges)
    denominator = lcm(*(value.denominator for value in fractions)) if fractions else 1
    if _decimal_digit_upper_bound(denominator) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_induced_weight.rational_height_bound",
            message=(
                "a common denominator for the induced-weight sums may exceed the "
                f"canonical {MAX_CANONICAL_RATIONAL_DIGITS:,}-digit rational bound"
            ),
        )

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    scaled = tuple(
        value.numerator * (denominator // value.denominator) for value in fractions
    )
    scaled_absolute_sum = sum(abs(entry) for entry in scaled)
    if _decimal_digit_upper_bound(scaled_absolute_sum) > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.signed_induced_weight.rational_height_bound",
            message=(
                "an induced-weight numerator may exceed the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit rational bound"
            ),
        )

    # Support components over nonzero-weight edges only.
    support_pairs = []
    support_touched = [False] * order
    for edge, weight in zip(graph.edges, scaled, strict=True):
        if weight == 0:
            continue
        left = vertex_index[edge.endpoints[0]]
        right = vertex_index[edge.endpoints[1]]
        support_touched[left] = support_touched[right] = True
        support_pairs.append((left, right))
    touched_groups = _support_groups(order, tuple(support_pairs))
    groups = tuple(
        group for group in touched_groups if any(support_touched[i] for i in group)
    )
    isolates = tuple(index for index in range(order) if not support_touched[index])

    maximum_bits = max(denominator.bit_length(), scaled_absolute_sum.bit_length(), 1)
    integer_limbs = (maximum_bits + 63) // 64
    presolve_work = len(graph.edges) + sum(len(edge.endpoints) for edge in graph.edges)
    scaled_triples = tuple(
        (vertex_index[edge.endpoints[0]], vertex_index[edge.endpoints[1]], weight)
        for edge, weight in zip(graph.edges, scaled, strict=True)
        if weight != 0
    )
    components: list[SignedWeightComponent] = []
    component_work = 0
    for group in groups:
        if len(group) > MAX_SIGNED_WEIGHT_COMPONENT_VERTICES:
            raise OperationDomainValidationError(
                location=("graph",),
                code="graph.signed_induced_weight.work_budget",
                message=(
                    "a nonzero-weight support component exceeds the "
                    f"{MAX_SIGNED_WEIGHT_COMPONENT_VERTICES}-vertex exhaustive "
                    "search envelope"
                ),
            )
        component = _charge_component(group, scaled_triples, integer_limbs)
        component_work += _component_work_unit(component, integer_limbs)
        components.append(component)

    # Witness reconstruction reuses constrained component searches: at most
    # (order + 1) tuple positions times (order + 1) candidates, each running
    # every component search once more. A single support component skips
    # this phase (its monolithic search already establishes the witness).
    greedy_work = 0
    if len(components) > 1:
        greedy_work = (order + 1) * (order + 1) * component_work
    work_units = presolve_work + component_work + greedy_work
    if work_units > MAX_SIGNED_WEIGHT_WORK_UNITS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.signed_induced_weight.work_budget",
            message=(
                "the component-bounded signed induced-weight search requires "
                f"{work_units:,} candidate and integer-limb update work units, "
                f"exceeding the {MAX_SIGNED_WEIGHT_WORK_UNITS:,}-unit bound"
            ),
        )

    return SignedInducedWeightAdmission(
        denominator=denominator,
        components=tuple(components),
        isolates=isolates,
        integer_limbs=integer_limbs,
        work_units=work_units,
    )


__all__ = [
    "MAX_SIGNED_WEIGHT_EDGES",
    "MAX_SIGNED_WEIGHT_VERTICES",
    "MAX_SIGNED_WEIGHT_WORK_UNITS",
    "MAX_SUBSET_ENUMERATION",
    "SignedInducedWeightAdmission",
    "admit_signed_induced_weight",
]
