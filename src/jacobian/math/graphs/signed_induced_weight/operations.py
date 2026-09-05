"""Exact signed induced-edge weight extrema over all vertex subsets."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.optimization._models import RationalWeightedGraph
from jacobian.math.graphs.signed_induced_weight._bounds import (
    SignedInducedWeightAdmission,
    SignedWeightComponent,
    admit_signed_induced_weight,
)
from jacobian.math.graphs.signed_induced_weight._models import (
    SignedInducedWeightResult,
    WeightExtremum,
)

__all__ = ["signed_induced_weight_extrema"]


def _gray_min_max(
    size: int,
    adjacency: tuple[tuple[tuple[int, int], ...], ...],
    *,
    bias: tuple[int, ...] | None = None,
    base: int = 0,
    track_witness: bool = False,
    labels: tuple[str, ...] | None = None,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Gray-code min/max over subsets with incremental scaled-integer updates.

    ``bias[x]`` is an extra linear term collected when ``x`` is selected, so
    constrained searches fix vertices by folding them into ``base``/``bias``.
    Returns ``(min_value, max_value, min_set, max_set)`` as local positions;
    the sets are meaningful only with ``track_witness``, breaking value ties
    by the lexicographically least label tuple exactly like the legacy
    monolithic kernel.
    """

    selected = [False] * size
    current = base
    minimum_value = base
    maximum_value = base
    min_set: tuple[int, ...] = ()
    max_set: tuple[int, ...] = ()
    # Empty-tuple keys reproduce the legacy kernel exactly: the empty witness
    # is the least tuple, so an all-zero optimum keeps witness ().
    min_key: tuple[str, ...] = ()
    max_key: tuple[str, ...] = ()
    previous_gray = 0
    bias = bias if bias is not None else (0,) * size

    for step in range(1, 1 << size):
        gray = step ^ (step >> 1)
        changed = (gray ^ previous_gray).bit_length() - 1
        if selected[changed]:
            selected[changed] = False
            current -= bias[changed] + sum(
                weight for neighbor, weight in adjacency[changed] if selected[neighbor]
            )
        else:
            current += bias[changed] + sum(
                weight for neighbor, weight in adjacency[changed] if selected[neighbor]
            )
            selected[changed] = True
        previous_gray = gray
        new_minimum = current < minimum_value
        new_maximum = current > maximum_value
        tied_minimum = tied_maximum = False
        witness: tuple[int, ...] = ()
        key: tuple[str, ...] = ()
        if track_witness and (
            new_minimum or new_maximum or current in (minimum_value, maximum_value)
        ):
            assert labels is not None
            witness = tuple(
                local for local, is_selected in enumerate(selected) if is_selected
            )
            key = tuple(labels[local] for local in witness)
            tied_minimum = current == minimum_value and key < min_key
            tied_maximum = current == maximum_value and key < max_key
        if new_minimum or tied_minimum:
            minimum_value = current
            if track_witness:
                min_set, min_key = witness, key
        if new_maximum or tied_maximum:
            maximum_value = current
            if track_witness:
                max_set, max_key = witness, key
    return minimum_value, maximum_value, min_set, max_set


def _scaled_edges(
    graph: RationalWeightedGraph, denominator: int
) -> tuple[tuple[int, int, int], ...]:
    """Return scaled integer edges as global-index triples."""

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    triples = []
    for edge in graph.edges:
        value = edge.weight.as_fraction()
        scaled = value.numerator * (denominator // value.denominator)
        triples.append(
            (vertex_index[edge.endpoints[0]], vertex_index[edge.endpoints[1]], scaled)
        )
    return tuple(triples)


def _declared_tuple(graph: RationalWeightedGraph, members: set[int]) -> tuple[str, ...]:
    return tuple(
        vertex for index, vertex in enumerate(graph.vertices) if index in members
    )


def _subset_value(
    chosen: set[int],
    scaled_edges: tuple[tuple[int, int, int], ...],
) -> int:
    return sum(
        weight
        for left, right, weight in scaled_edges
        if left in chosen and right in chosen
    )


def _constrained_component_optimum(
    component: SignedWeightComponent,
    forced_in: frozenset[int],
    forced_out: frozenset[int],
    *,
    find_min: bool,
) -> int:
    """Best scaled value inside one component under fixed global vertices.

    Fixed-in vertices fold into the base value and per-vertex biases; the
    Gray-code search runs over the remaining free vertices only. Zero-weight
    edges never enter component adjacency and contribute nothing.
    """

    size = len(component.vertices)
    base = 0
    bias = [0] * size
    free: list[int] = []
    for local in range(len(component.vertices)):
        global_index = component.vertices[local]
        if global_index in forced_out:
            continue
        if global_index in forced_in:
            # Each forced-in/forced-in edge counts once; each
            # forced-in/free edge becomes a linear bias on the free end.
            # (Free endpoints add no bias themselves: their forced
            # neighbors already contributed it above.)
            for neighbor, weight in component.adjacency[local]:
                other = component.vertices[neighbor]
                if other in forced_in:
                    if neighbor > local:
                        base += weight
                elif other not in forced_out:
                    bias[neighbor] += weight
            continue
        free.append(local)
    free_position = {local: rank for rank, local in enumerate(free)}
    free_adjacency = tuple(
        tuple(
            (free_position[neighbor], weight)
            for neighbor, weight in component.adjacency[local]
            if neighbor in free_position
        )
        for local in free
    )
    free_bias = tuple(bias[local] for local in free)
    minimum, maximum, _, _ = _gray_min_max(
        len(free), free_adjacency, bias=free_bias, base=base
    )
    return minimum if find_min else maximum


def _feasible_optimum(
    admission: SignedInducedWeightAdmission,
    forced_in: frozenset[int],
    forced_out: frozenset[int],
    *,
    find_min: bool,
) -> int:
    """Best scaled value over supersets of ``forced_in`` avoiding ``forced_out``."""

    return sum(
        _constrained_component_optimum(
            component, forced_in, forced_out, find_min=find_min
        )
        for component in admission.components
    )


def _lex_least_achiever(
    graph: RationalWeightedGraph,
    admission: SignedInducedWeightAdmission,
    scaled_edges: tuple[tuple[int, int, int], ...],
    target: int,
    *,
    find_min: bool,
) -> tuple[str, ...]:
    """Return the lexicographically least vertex tuple attaining ``target``.

    Tuple positions are filled left to right over declared slots: each
    position takes the smallest label with a feasible completion, where
    feasibility is an exact constrained component optimization. A set
    already attaining ``target`` stops the construction since extensions
    append declared-later vertices.
    """

    if target == 0:
        return ()
    chosen: set[int] = set()
    last_position = -1
    order = len(graph.vertices)
    while True:
        prefix_forbidden = {
            index for index in range(last_position + 1) if index not in chosen
        }
        if _subset_value(chosen, scaled_edges) == target:
            return _declared_tuple(graph, chosen)
        candidates = sorted(
            graph.vertices[index]
            for index in range(last_position + 1, order)
            if index not in chosen
        )
        picked: int | None = None
        for label in candidates:
            index = graph.vertices.index(label)
            if (
                _feasible_optimum(
                    admission,
                    frozenset(chosen | {index}),
                    frozenset(prefix_forbidden),
                    find_min=find_min,
                )
                == target
            ):
                picked = index
                break
        if picked is None:  # pragma: no cover - admission keeps a pick available
            raise AssertionError("lexicographic witness search exhausted candidates")
        chosen.add(picked)
        last_position = picked


def _with_free_isolates(
    graph: RationalWeightedGraph,
    witness: tuple[str, ...],
    isolates: tuple[int, ...],
) -> tuple[str, ...]:
    """Extend a component witness with value-neutral isolated vertices.

    Isolated vertices (no nonzero-weight edges) never change the induced
    weight, so the lexicographically least optimum adds each of them in
    ascending label order exactly when it shrinks the declared-order tuple.
    """

    chosen = {graph.vertices.index(vertex) for vertex in witness}
    for label in sorted(graph.vertices[index] for index in isolates):
        index = graph.vertices.index(label)
        if index in chosen:
            continue
        candidate = _declared_tuple(graph, chosen | {index})
        if candidate < _declared_tuple(graph, chosen):
            chosen.add(index)
    return _declared_tuple(graph, chosen)


def _extremum_witnesses(
    graph: RationalWeightedGraph,
    admission: SignedInducedWeightAdmission,
    scaled_edges: tuple[tuple[int, int, int], ...],
    minimum_value: int,
    maximum_value: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return exact lex-least (min, max) witnesses for established optima."""

    if len(admission.components) <= 1:
        if not admission.components:
            return (), ()
        (component,) = admission.components
        labels = tuple(graph.vertices[index] for index in component.vertices)
        _, _, min_local, max_local = _gray_min_max(
            len(component.vertices),
            component.adjacency,
            track_witness=True,
            labels=labels,
        )
        min_witness = _with_free_isolates(
            graph,
            _declared_tuple(graph, {component.vertices[local] for local in min_local}),
            admission.isolates,
        )
        max_witness = _with_free_isolates(
            graph,
            _declared_tuple(graph, {component.vertices[local] for local in max_local}),
            admission.isolates,
        )
        return min_witness, max_witness
    return (
        _lex_least_achiever(
            graph, admission, scaled_edges, minimum_value, find_min=True
        ),
        _lex_least_achiever(
            graph, admission, scaled_edges, maximum_value, find_min=False
        ),
    )


def signed_induced_weight_extrema(
    graph: RationalWeightedGraph,
) -> SignedInducedWeightResult:
    """Return exact min and max signed induced-edge weight over all subsets.

    For a vertex subset S, the value is
    ``sum(weight(u,v) : {u,v} is an edge and both endpoints are in S)``.
    Empty and singleton subsets have weight zero. Ties are broken by the
    lexicographically least selected vertex axis. Nonzero-weight support
    components are optimized independently and charged together, so sparse
    graphs beyond the monolithic vertex envelope are admitted when every
    component fits it.
    """
    admission = admit_signed_induced_weight(graph)
    scaled_edges = _scaled_edges(graph, admission.denominator)
    minimum_value = sum(
        _gray_min_max(len(component.vertices), component.adjacency)[0]
        for component in admission.components
    )
    maximum_value = sum(
        _gray_min_max(len(component.vertices), component.adjacency)[1]
        for component in admission.components
    )
    min_witness, max_witness = _extremum_witnesses(
        graph, admission, scaled_edges, minimum_value, maximum_value
    )
    return SignedInducedWeightResult(
        graph=graph,
        minimum=WeightExtremum(
            value=CanonicalRational.from_fraction(
                Fraction(minimum_value, admission.denominator)
            ),
            witness_vertices=min_witness,
        ),
        maximum=WeightExtremum(
            value=CanonicalRational.from_fraction(
                Fraction(maximum_value, admission.denominator)
            ),
            witness_vertices=max_witness,
        ),
    )
