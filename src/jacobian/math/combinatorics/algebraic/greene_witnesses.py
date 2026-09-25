"""Exact bounded disjoint subsequence witnesses for Greene invariants."""

from __future__ import annotations

from math import inf

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.algebraic._rsk import _row_insert
from jacobian.math.combinatorics.algebraic.biword import (
    MAX_GREENE_WITNESS_WORD_LENGTH,
    GreeneWitnessFamily,
    GreeneWitnessRequest,
    GreeneWitnessResult,
)
from jacobian.math.combinatorics.symmetric_functions.values import IntegerPartition

_MAX_RELAXATIONS = 6_000_000
_MAX_OUTPUT_MEMBERSHIPS = 512

type _Edge = list[int]


def _add_edge(graph: list[list[_Edge]], source: int, target: int, cost: int) -> int:
    """Add one unit-capacity residual edge and return its forward edge index."""

    forward_index = len(graph[source])
    reverse_index = len(graph[target])
    graph[source].append([target, reverse_index, 1, cost])
    graph[target].append([source, forward_index, 0, -cost])
    return forward_index


def _maximum_weight_disjoint_chains(
    ranks: tuple[int, ...], path_count: int, *, weakly_increasing: bool
) -> tuple[tuple[int, ...], ...]:
    """Find a maximum-cardinality union of at most ``path_count`` chains.

    A unit-capacity min-cost flow sends one path through each selected source
    position. Entering a position costs -1; transition arcs encode the chosen
    strict or weak subsequence order. Successive shortest augmenting paths are
    found with Bellman-Ford because the residual graph contains negative-cost
    reverse arcs.
    """

    n = len(ranks)
    if not n or not path_count:
        return ()

    graph, source, sink, source_edges, vertex_edges, transition_edges = (
        _build_chain_network(ranks, weakly_increasing=weakly_increasing)
    )
    _send_min_cost_flow(graph, source, sink, path_count)
    return _decompose_flow_paths(
        graph, source_edges, vertex_edges, transition_edges, path_count
    )


def _build_chain_network(
    ranks: tuple[int, ...], *, weakly_increasing: bool
) -> tuple[
    list[list[_Edge]],
    int,
    int,
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int, int, int]],
]:
    n = len(ranks)
    source, sink = 2 * n, 2 * n + 1
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]
    source_edges: list[tuple[int, int]] = []
    vertex_edges: list[tuple[int, int]] = []
    transition_edges: list[tuple[int, int, int, int]] = []
    for index in range(n):
        entry = 2 * index
        leave = entry + 1
        source_index = _add_edge(graph, source, entry, 0)
        vertex_index = _add_edge(graph, entry, leave, -1)
        _add_edge(graph, leave, sink, 0)
        source_edges.append((source, source_index))
        vertex_edges.append((entry, vertex_index))
    for left in range(n):
        for right in range(left + 1, n):
            compatible = (
                ranks[left] <= ranks[right]
                if weakly_increasing
                else ranks[left] > ranks[right]
            )
            if compatible:
                node = 2 * left + 1
                edge_index = _add_edge(graph, node, 2 * right, 0)
                transition_edges.append((left, right, node, edge_index))
    return graph, source, sink, source_edges, vertex_edges, transition_edges


def _send_min_cost_flow(
    graph: list[list[_Edge]], source: int, sink: int, path_count: int
) -> None:
    for _unit in range(path_count):
        distances: list[int | float] = [inf] * len(graph)
        parents: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0
        for _pass in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                base = distances[node]
                if base == inf:
                    continue
                for edge_index, (target, _reverse, capacity, cost) in enumerate(edges):
                    if capacity and base + cost < distances[target]:
                        distances[target] = base + cost
                        parents[target] = (node, edge_index)
                        changed = True
            request_checkpoint("during Greene witness shortest path")
            if not changed:
                break
        if parents[sink] is None:
            raise RuntimeError("the disjoint singleton path flow became infeasible")
        node = sink
        while node != source:
            parent = parents[node]
            if parent is None:
                raise RuntimeError("shortest-path predecessor chain is incomplete")
            previous, edge_index = parent
            edge = graph[previous][edge_index]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = previous


def _decompose_flow_paths(
    graph: list[list[_Edge]],
    source_edges: list[tuple[int, int]],
    vertex_edges: list[tuple[int, int]],
    transition_edges: list[tuple[int, int, int, int]],
    path_count: int,
) -> tuple[tuple[int, ...], ...]:
    selected = {
        index
        for index, (node, edge) in enumerate(vertex_edges)
        if graph[node][edge][2] == 0
    }
    starts = [
        index
        for index, (node, edge) in enumerate(source_edges)
        if graph[node][edge][2] == 0
    ]
    successor = {
        left: right
        for left, right, node, edge in transition_edges
        if graph[node][edge][2] == 0
    }
    paths: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for start in starts:
        path: list[int] = []
        current = start
        while True:
            if current in visited or current not in selected:
                raise RuntimeError("flow did not decompose into disjoint paths")
            visited.add(current)
            path.append(current)
            if current not in successor:
                break
            current = successor[current]
        paths.append(tuple(path))
    if visited != selected or len(paths) != path_count:
        raise RuntimeError("flow path decomposition lost selected positions")
    return tuple(paths)


def _maximum_relaxation_bound(n: int, k: int) -> int:
    vertices = 2 * n + 2
    # At most n(n-1)/2 order arcs plus 3n source, vertex, and sink arcs;
    # every forward arc has one residual reverse arc.
    residual_arcs = n * (n - 1) + 6 * n
    augmentations = sum(min(index, n) for index in range(1, k + 1))
    return 2 * augmentations * (vertices - 1) * residual_arcs


def compute_greene_witnesses(
    request: GreeneWitnessRequest,
) -> GreeneWitnessResult:
    """Return disjoint weak-increasing and strict-decreasing Greene witnesses."""

    if type(request) is not GreeneWitnessRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="algebraic_combinatorics.greene_witness_request",
            message="expected a canonical Greene witness request",
        )
    try:
        request = GreeneWitnessRequest.model_validate(request.model_dump(mode="python"))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="algebraic_combinatorics.greene_witness_request",
            message="Greene witness request is malformed",
        ) from exc

    word = request.word
    n = len(word.letters)
    k = request.k
    if n > MAX_GREENE_WITNESS_WORD_LENGTH:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.greene_witness_word_limit",
            message=(
                "disjoint Greene witnesses are bounded to words of at most "
                f"{MAX_GREENE_WITNESS_WORD_LENGTH} letters"
            ),
        )
    output_memberships = 2 * k * n
    work = _maximum_relaxation_bound(n, k) + n * n + output_memberships
    if work > _MAX_RELAXATIONS:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.greene_witness_work",
            message="disjoint Greene witness search exceeds the admitted work bound",
        )
    if output_memberships > _MAX_OUTPUT_MEMBERSHIPS:
        raise OperationResourceAdmissionError(
            location=("k",),
            code="algebraic_combinatorics.greene_witness_output",
            message="Greene witness families exceed the aggregate output bound",
        )

    rank_by_symbol = {symbol: index for index, symbol in enumerate(word.alphabet)}
    ranks = tuple(rank_by_symbol[letter] for letter in word.letters)
    insertion, _ = _row_insert(tuple(rank + 1 for rank in ranks))
    shape = IntegerPartition(parts=tuple(len(row) for row in insertion))
    families: list[GreeneWitnessFamily] = []
    for index in range(1, k + 1):
        path_count = min(index, n)
        inc_paths = _maximum_weight_disjoint_chains(
            ranks, path_count, weakly_increasing=True
        )
        dec_paths = _maximum_weight_disjoint_chains(
            ranks, path_count, weakly_increasing=False
        )
        families.append(
            GreeneWitnessFamily(
                k=index,
                increasing_subsequences=inc_paths,
                decreasing_subsequences=dec_paths,
                increasing_total=sum(map(len, inc_paths)),
                decreasing_total=sum(map(len, dec_paths)),
            )
        )
    return GreeneWitnessResult._from_kernel(
        word=word,
        shape=shape,
        families=tuple(families),
    )


__all__ = ["compute_greene_witnesses"]
