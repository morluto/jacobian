"""Exact all-terminal reliability for bounded simple graphs."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from functools import cache
from math import comb
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    DecimalIntegerEncoding,
    require_bounded_rational,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import SimpleUndirectedGraph

if TYPE_CHECKING:
    import networkx as nx

MAX_ALL_TERMINAL_RELIABILITY_EDGES = 20
MAX_ALL_TERMINAL_RELIABILITY_STATES = 1 << MAX_ALL_TERMINAL_RELIABILITY_EDGES
MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS = 128
# For p=a/b, the reduced reliability denominator divides b**m and its
# numerator is no larger because the result is a probability.
MAX_ALL_TERMINAL_RELIABILITY_RESULT_DIGITS = (
    MAX_ALL_TERMINAL_RELIABILITY_EDGES * MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS + 1
)
_MAX_ALL_TERMINAL_RELIABILITY_INPUT_ABS = 10**MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS
_MAX_ALL_TERMINAL_RELIABILITY_RESULT_ABS = (
    10**MAX_ALL_TERMINAL_RELIABILITY_RESULT_DIGITS
)


def _require_bounded_problem(
    graph: SimpleUndirectedGraph,
    open_probability: Fraction,
) -> None:
    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("graph must be a SimpleUndirectedGraph")
    vertex_count = len(graph.vertices)
    edge_count = len(graph.edges)
    if vertex_count == 0:
        raise ValueError("all-terminal reliability requires a nonempty graph")
    if edge_count > MAX_ALL_TERMINAL_RELIABILITY_EDGES:
        raise ValueError(
            "all-terminal reliability exceeds the "
            f"{MAX_ALL_TERMINAL_RELIABILITY_EDGES}-edge bound"
        )
    if type(open_probability) is not Fraction:
        raise TypeError("open_probability must be a Fraction")
    if not 0 <= open_probability <= 1:
        raise ValueError("open_probability must lie in [0, 1]")
    if (
        abs(open_probability.numerator) >= _MAX_ALL_TERMINAL_RELIABILITY_INPUT_ABS
        or open_probability.denominator >= _MAX_ALL_TERMINAL_RELIABILITY_INPUT_ABS
    ):
        raise ValueError(
            "open_probability exceeds the "
            f"{MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS}-digit bound"
        )


def _indexed_graph(
    graph: SimpleUndirectedGraph,
) -> tuple[nx.Graph[int], tuple[tuple[int, int], ...]]:
    import networkx as nx

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    indexed_edges = tuple(
        (vertex_index[left], vertex_index[right]) for left, right in graph.edges
    )
    backend_graph: nx.Graph[int] = nx.Graph()
    backend_graph.add_nodes_from(range(len(graph.vertices)))
    return backend_graph, indexed_edges


def _connected_spanning_subgraph_counts(
    graph: SimpleUndirectedGraph,
) -> tuple[int, ...]:
    """Count connected spanning edge subsets, indexed by subset cardinality."""

    import networkx as nx

    backend_graph, edges = _indexed_graph(graph)
    counts = [0] * (len(edges) + 1)

    previous_gray_code = 0
    open_edge_count = 0
    for state_index in range(1 << len(edges)):
        gray_code = state_index ^ (state_index >> 1)
        if state_index:
            changed_bit = gray_code ^ previous_gray_code
            edge_index = changed_bit.bit_length() - 1
            edge = edges[edge_index]
            if gray_code & changed_bit:
                backend_graph.add_edge(*edge)
                open_edge_count += 1
            else:
                backend_graph.remove_edge(*edge)
                open_edge_count -= 1
        if open_edge_count >= len(graph.vertices) - 1 and nx.is_connected(
            backend_graph
        ):
            counts[open_edge_count] += 1
        previous_gray_code = gray_code
    return tuple(counts)


_InternalGraph = tuple[int, tuple[tuple[int, int], ...]]


def _canonical_internal_graph(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
) -> _InternalGraph:
    """Canonicalize the multigraph used by deletion/contraction.

    Edges are retained with multiplicity.  A loop is represented by ``(v, v)``
    and is not discarded: contraction can create loops, whose two states each
    contribute to the edge-count profile.
    """

    return vertex_count, tuple(
        sorted((min(left, right), max(left, right)) for left, right in edges)
    )


def _internal_graph_is_connected(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
) -> bool:
    if vertex_count <= 1:
        return True
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    for left, right in edges:
        if left != right:
            adjacency[left].append(right)
            adjacency[right].append(left)
    seen = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return len(seen) == vertex_count


def _internal_bridge_indices(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    """Return bridge edge indices, handling parallel edges by edge identity."""

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for edge_index, (left, right) in enumerate(edges):
        if left != right:
            adjacency[left].append((right, edge_index))
            adjacency[right].append((left, edge_index))
    discovery = [-1] * vertex_count
    low = [-1] * vertex_count
    bridges: list[int] = []
    clock = 0
    # Keep the DFS iterative because the admitted graph carrier permits more
    # than ten thousand vertices, while an edge-sparse path can be that deep.
    for root in range(vertex_count):
        if discovery[root] != -1:
            continue
        discovery[root] = low[root] = clock
        clock += 1
        stack: list[tuple[int, int, int, int]] = [(root, -1, -1, 0)]
        while stack:
            vertex, parent, parent_edge, next_index = stack[-1]
            if next_index == len(adjacency[vertex]):
                stack.pop()
                if parent != -1:
                    low[parent] = min(low[parent], low[vertex])
                    if low[vertex] > discovery[parent]:
                        bridges.append(parent_edge)
                continue
            neighbor, edge_index = adjacency[vertex][next_index]
            stack[-1] = (vertex, parent, parent_edge, next_index + 1)
            if edge_index == parent_edge:
                continue
            if discovery[neighbor] == -1:
                discovery[neighbor] = low[neighbor] = clock
                clock += 1
                stack.append((neighbor, vertex, edge_index, 0))
            else:
                low[vertex] = min(low[vertex], discovery[neighbor])
    return tuple(sorted(bridges))


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            parent = self.parent
            parent[right_root] = left_root


def _contract_bridges(
    graph: _InternalGraph,
    bridges: tuple[int, ...],
) -> tuple[_InternalGraph, int]:
    vertex_count, edges = graph
    bridge_set = set(bridges)
    components = _DisjointSet(vertex_count)
    for edge_index in bridges:
        left, right = edges[edge_index]
        components.union(left, right)
    roots = sorted({components.find(vertex) for vertex in range(vertex_count)})
    root_index = {root: index for index, root in enumerate(roots)}
    contracted_edges = tuple(
        (
            root_index[components.find(left)],
            root_index[components.find(right)],
        )
        for edge_index, (left, right) in enumerate(edges)
        if edge_index not in bridge_set
    )
    return (
        _canonical_internal_graph(len(roots), contracted_edges),
        len(bridges),
    )


def _delete_internal_edge(graph: _InternalGraph, edge_index: int) -> _InternalGraph:
    vertex_count, edges = graph
    return _canonical_internal_graph(
        vertex_count,
        tuple(edge for index, edge in enumerate(edges) if index != edge_index),
    )


def _contract_internal_edge(graph: _InternalGraph, edge_index: int) -> _InternalGraph:
    vertex_count, edges = graph
    left, right = edges[edge_index]
    if left == right:
        raise ValueError("deletion/contraction requires a non-loop edge")

    # The edge tuple is canonical, so right is the vertex removed from the axis.
    def remap(vertex: int) -> int:
        return vertex - (vertex > right)

    contracted_edges = []
    for index, (edge_left, edge_right) in enumerate(edges):
        if index == edge_index:
            continue
        edge_left = left if edge_left == right else edge_left
        edge_right = left if edge_right == right else edge_right
        contracted_edges.append((remap(edge_left), remap(edge_right)))
    return _canonical_internal_graph(vertex_count - 1, tuple(contracted_edges))


def _polynomial_shift(polynomial: tuple[int, ...], amount: int) -> tuple[int, ...]:
    return (0,) * amount + polynomial


def _polynomial_add(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    size = max(len(left), len(right))
    return tuple(
        (left[index] if index < len(left) else 0)
        + (right[index] if index < len(right) else 0)
        for index in range(size)
    )


def _all_loop_subsets(loop_count: int) -> tuple[int, ...]:
    return tuple(comb(loop_count, index) for index in range(loop_count + 1))


def _deletion_contraction_profile(
    graph: _InternalGraph,
) -> tuple[tuple[int, ...], int]:
    """Compute ``C_G(z)`` and the number of unique recursive states.

    The recurrence is ``C_G = C_{G-e} + z C_{G/e}`` for a non-loop edge.
    Bridge components are contracted together before branching; all such edges
    are mandatory and therefore shift the profile by their count.
    """

    @cache
    def solve(state: _InternalGraph) -> tuple[int, ...]:
        vertex_count, edges = state
        if vertex_count == 1:
            return _all_loop_subsets(len(edges))
        if not _internal_graph_is_connected(vertex_count, edges):
            return (0,)

        bridges = _internal_bridge_indices(vertex_count, edges)
        if bridges:
            reduced, bridge_count = _contract_bridges(state, bridges)
            return _polynomial_shift(solve(reduced), bridge_count)

        edge_index = next(
            (index for index, (left, right) in enumerate(edges) if left != right),
            None,
        )
        if edge_index is None:
            return (0,)
        deleted = solve(_delete_internal_edge(state, edge_index))
        contracted = solve(_contract_internal_edge(state, edge_index))
        return _polynomial_add(deleted, _polynomial_shift(contracted, 1))

    profile = solve(graph)
    return profile, solve.cache_info().currsize


def _connected_spanning_subgraph_counts_with_work(
    graph: SimpleUndirectedGraph,
) -> tuple[tuple[int, ...], int]:
    """Select the exact kernel and report its actual explored-state count."""

    vertex_count = len(graph.vertices)
    edge_count = len(graph.edges)
    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    indexed_edges = tuple(
        (vertex_index[left], vertex_index[right]) for left, right in graph.edges
    )
    internal = _canonical_internal_graph(vertex_count, indexed_edges)
    # The recursive kernel is materially cheaper once the profile has at least
    # eight edge axes.  Keep the small exhaustive regime stable and explicit.
    if edge_count >= 8:
        profile, work = _deletion_contraction_profile(internal)
        return profile + (0,) * (edge_count + 1 - len(profile)), work
    bridge_indices = _internal_bridge_indices(vertex_count, internal[1])
    if not _internal_graph_is_connected(vertex_count, internal[1]) or bridge_indices:
        profile, work = _deletion_contraction_profile(internal)
        return profile + (0,) * (edge_count + 1 - len(profile)), work
    return _connected_spanning_subgraph_counts(graph), 1 << edge_count


def _evaluate_reliability(
    counts: tuple[int, ...],
    open_probability: Fraction,
) -> Fraction:
    closed_probability = 1 - open_probability
    edge_count = len(counts) - 1
    return sum(
        (
            count
            * open_probability**open_edges
            * closed_probability ** (edge_count - open_edges)
            for open_edges, count in enumerate(counts)
        ),
        Fraction(),
    )


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.reliability_invariant", message)


class AllTerminalReliabilityResult(StrictModel):
    """Exact probability with its bounded connected-subgraph profile.

    Deserialization checks the structural result envelope. The kernel uses
    ``_from_kernel`` after one exact profile computation.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Exact all-terminal reliability bound to the retained graph and "
                "uniform edge probability. Entry k of "
                "`connected_spanning_subgraph_counts` is the number of connected "
                "spanning edge subsets containing exactly k edges."
            )
        }
    )

    graph: SimpleUndirectedGraph
    open_probability: CanonicalRational
    connected_spanning_subgraph_counts: tuple[
        Annotated[int, DecimalIntegerEncoding(max_digits=7)], ...
    ] = Field(
        min_length=1,
        max_length=MAX_ALL_TERMINAL_RELIABILITY_EDGES + 1,
        description=(
            "Counts indexed by open-edge cardinality k=0..m. They reconstruct "
            "R_G(p)=sum_k c_k p^k (1-p)^(m-k)."
        ),
    )
    reliability_probability: CanonicalRational
    visited_states: StrictInt = Field(
        ge=1,
        le=MAX_ALL_TERMINAL_RELIABILITY_STATES,
        description=(
            "The number of unique edge-subset or deletion/contraction states "
            "explored by the exact kernel; it is at most 2^m."
        ),
    )
    event: Literal["ALL_VERTICES_CONNECTED"] = "ALL_VERTICES_CONNECTED"

    @model_validator(mode="before")
    @classmethod
    def bound_raw_coefficients(cls, value: Any) -> Any:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        raw_counts = value.get("connected_spanning_subgraph_counts")
        if isinstance(raw_counts, (list, tuple)):
            if len(raw_counts) > MAX_ALL_TERMINAL_RELIABILITY_EDGES + 1:
                raise _validation_error(
                    "connected-spanning-subgraph profile exceeds the edge bound"
                )
            max_digits = len(str(MAX_ALL_TERMINAL_RELIABILITY_STATES))
            if any(
                isinstance(item, str) and len(item.lstrip("-")) > max_digits
                for item in raw_counts
            ):
                raise _validation_error(
                    "connected-spanning-subgraph count exceeds the state bound"
                )
        return value

    @model_validator(mode="after")
    def require_bounded_shape(self) -> Self:
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS,
            label="all-terminal reliability open probability",
        )
        require_bounded_rational(
            self.reliability_probability,
            max_digits=MAX_ALL_TERMINAL_RELIABILITY_RESULT_DIGITS,
            label="all-terminal reliability result probability",
        )
        if not 0 <= self.open_probability.as_fraction() <= 1:
            raise _validation_error(
                "all-terminal reliability open probability must lie in [0, 1]"
            )
        actual_counts = tuple(
            value for value in self.connected_spanning_subgraph_counts
        )
        if len(actual_counts) != len(self.graph.edges) + 1:
            raise _validation_error(
                "connected-spanning-subgraph counts must cover edge counts 0..m"
            )
        if any(
            count < 0 or count > MAX_ALL_TERMINAL_RELIABILITY_STATES
            for count in actual_counts
        ):
            raise _validation_error(
                "connected-spanning-subgraph count exceeds its bounded integer range"
            )
        if self.visited_states > 1 << len(self.graph.edges):
            raise _validation_error("visited_states exceeds the complete edge powerset")
        if not 0 <= self.reliability_probability.as_fraction() <= 1:
            raise _validation_error(
                "all-terminal reliability result probability must lie in [0, 1]"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        graph: SimpleUndirectedGraph,
        open_probability: Fraction,
        counts: tuple[int, ...],
        reliability_probability: Fraction,
        visited_states: int,
    ) -> Self:
        """Build trusted kernel output without replaying the enumeration."""

        return cls.model_construct(
            graph=graph,
            open_probability=CanonicalRational.from_fraction(open_probability),
            connected_spanning_subgraph_counts=tuple(count for count in counts),
            reliability_probability=CanonicalRational.from_fraction(
                reliability_probability
            ),
            visited_states=visited_states,
            event="ALL_VERTICES_CONNECTED",
        )


def _compute_all_terminal_reliability(
    graph: SimpleUndirectedGraph,
    open_probability: Fraction,
) -> tuple[tuple[int, ...], Fraction, int]:
    try:
        _require_bounded_problem(graph, open_probability)
    except (TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("graph", "open_probability"),
            code="probability.all_terminal_reliability_not_admitted",
            message=str(exc),
        ) from None
    counts, visited_states = _connected_spanning_subgraph_counts_with_work(graph)
    return (
        counts,
        _evaluate_reliability(counts, open_probability),
        visited_states,
    )


def all_terminal_reliability(
    graph: SimpleUndirectedGraph,
    open_probability: Fraction,
) -> AllTerminalReliabilityResult:
    """Compute exact uniform-edge all-terminal reliability.

    The coefficient at index ``k`` counts spanning connected subgraphs with
    exactly ``k`` open edges. The probability is
    ``sum(c[k] * p**k * (1-p)**(m-k))``.
    """

    counts, reliability_probability, visited_states = _compute_all_terminal_reliability(
        graph, open_probability
    )
    return AllTerminalReliabilityResult._from_kernel(
        graph,
        open_probability,
        counts,
        reliability_probability,
        visited_states,
    )


__all__ = [
    "AllTerminalReliabilityResult",
    "all_terminal_reliability",
]
