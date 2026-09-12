"""Cycle-length profile kernel."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from itertools import pairwise

import networkx as nx

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs._networkx import biconnected_components
from jacobian.math.graphs.cycle_length_profile._models import (
    MAX_VERTICES,
    CycleFamilyKind,
    CycleIncidenceRow,
    CycleLengthProfileResult,
    CycleLengthRow,
    FixedLengthCycleEnumerationResult,
    dihedral_canonical_cycle,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = [
    "MAX_FIXED_CYCLES",
    "MAX_FIXED_CYCLE_WORK",
    "compute_cycle_length_profile",
    "enumerate_chordless_fixed_length_cycles",
    "enumerate_fixed_length_cycles",
    "verify_cycle_length_profile",
    "verify_cycle_length_row",
]

MAX_SEARCH_WORK = 10_000_000
MAX_CYCLE_PROFILE_RETAINED_LABEL_CHARACTERS = 100_000_000
MAX_FIXED_CYCLE_WORK = 10_000_000
MAX_FIXED_CYCLES = 20_000
MAX_FIXED_CYCLE_RETAINED_LABEL_CHARACTERS = 100_000_000


@dataclass(frozen=True, slots=True)
class _BlockPlan:
    graph: SimpleUndirectedGraph
    wheel_order: tuple[str, ...] | None = None
    cycle_order: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _AdmissionPlan:
    graph: SimpleUndirectedGraph
    blocks: tuple[_BlockPlan, ...]


def _maximum_path_work(graph: SimpleUndirectedGraph, *, is_wheel: bool = False) -> int:
    """Count simple-path prefixes with an early work cutoff."""
    vertex_count = len(graph.vertices)
    if len(graph.edges) == vertex_count * (vertex_count - 1) // 2:
        # In a complete graph the first DFS branch witnesses every length.
        return vertex_count**3
    if is_wheel:
        # A wheel is pancyclic: the hub plus a contiguous rim segment gives
        # every length from three through n.  The kernel's first-witness DFS
        # reaches each such segment after at most O(n^2) neighbor checks, so
        # charge one cubic envelope instead of all simple paths.
        return vertex_count**3
    vertex_to_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    adjacency: list[list[bool]] = [[False] * vertex_count for _ in range(vertex_count)]
    for left, right in graph.edges:
        left_index = vertex_to_index[left]
        right_index = vertex_to_index[right]
        adjacency[left_index][right_index] = True
        adjacency[right_index][left_index] = True

    # The kernel repeats root and one-edge scans for every target length, even
    # when no path reaches depth three (for example, a perfect matching).
    root_scan_candidates = vertex_count * (vertex_count - 1) // 2 + 2 * len(graph.edges)
    work = root_scan_candidates * max(1, vertex_count - 2)
    # Each root edge accepted by ``nxt >= start + 1`` causes the kernel to
    # scan the complete remaining suffix once for every target length, even
    # when that edge has no continuation. Charge this depth-two work instead
    # of treating every edge as a constant-cost branch.
    depth_two_candidates = 0
    for left, right in graph.edges:
        left_index = vertex_to_index[left]
        right_index = vertex_to_index[right]
        depth_two_candidates += vertex_count - min(left_index, right_index) - 1
    work += depth_two_candidates * max(1, vertex_count - 2)
    for start in range(vertex_count):
        visited = {start}

        def visit(current: int, visited: set[int] = visited) -> None:
            nonlocal work
            for nxt in range(vertex_count):
                if nxt in visited or not adjacency[current][nxt]:
                    continue
                visited.add(nxt)
                if len(visited) >= 3:
                    work += vertex_count * vertex_count
                    if work > MAX_SEARCH_WORK:
                        return
                visit(nxt)
                visited.remove(nxt)
                if work > MAX_SEARCH_WORK:
                    return

        visit(start)
        if work > MAX_SEARCH_WORK:
            return work
    return work


def _wheel_search_order(graph: SimpleUndirectedGraph) -> tuple[str, ...] | None:
    """Return a hub-then-cyclic-rim order for the wheel DFS shortcut.

    Recognizes the wheel topology and returns the cyclic rim order in one pass,
    so callers that need both recognition and the order do not re-derive it.
    """
    vertex_count = len(graph.vertices)
    if vertex_count < 4 or len(graph.edges) != 2 * (vertex_count - 1):
        return None
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    hubs = [
        vertex
        for vertex, neighbors in adjacency.items()
        if len(neighbors) == vertex_count - 1
    ]
    if len(hubs) != 1:
        return None
    rim = [vertex for vertex in graph.vertices if vertex != hubs[0]]
    if not all(len(adjacency[vertex]) == 3 for vertex in rim):
        return None
    rim_adjacency = {vertex: adjacency[vertex] - {hubs[0]} for vertex in rim}
    if not all(len(neighbors) == 2 for neighbors in rim_adjacency.values()):
        return None
    seen = {rim[0]}
    stack = [rim[0]]
    while stack:
        vertex = stack.pop()
        for neighbor in rim_adjacency[vertex] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    if len(seen) != len(rim):
        return None
    hub = next(
        vertex
        for vertex, neighbors in adjacency.items()
        if len(neighbors) == len(graph.vertices) - 1
    )
    rim = sorted(vertex for vertex in graph.vertices if vertex != hub)
    rim_adjacency = {vertex: adjacency[vertex] - {hub} for vertex in rim}
    order = [rim[0]]
    previous: str | None = None
    current = rim[0]
    while len(order) < len(rim):
        candidates = sorted(
            neighbor for neighbor in rim_adjacency[current] if neighbor != previous
        )
        next_vertex = candidates[0] if candidates[0] not in order else candidates[1]
        order.append(next_vertex)
        previous, current = current, next_vertex
    return (hub, *order)


def _cycle_core_vertices(graph: SimpleUndirectedGraph) -> set[str]:
    """Return vertices in the graph's cycle-bearing 2-core."""
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    pending = [vertex for vertex, neighbors in adjacency.items() if len(neighbors) < 2]
    removed = set(pending)
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor in removed:
                continue
            adjacency[neighbor].discard(vertex)
            if len(adjacency[neighbor]) < 2:
                removed.add(neighbor)
                pending.append(neighbor)
    remaining = set(adjacency) - removed
    core_vertices: set[str] = set()
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        core_vertices.update(component)
    return core_vertices


def _reject(code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=("graph",), code=code, message=message
    )


def _admit(graph: SimpleUndirectedGraph) -> _AdmissionPlan:
    """Validate native and MCP graph, work, and result envelopes once."""
    if not isinstance(graph, SimpleUndirectedGraph):
        _reject(
            "cycle_profile.graph_type",
            "graph must be a canonical simple undirected graph",
        )
    vertex_count = len(graph.vertices)
    if vertex_count > MAX_VERTICES:
        _reject(
            "cycle_profile.vertex_bound",
            f"cycle profiles support at most {MAX_VERTICES} vertices",
        )
    retained_label_characters = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    if retained_label_characters > MAX_CYCLE_PROFILE_RETAINED_LABEL_CHARACTERS:
        _reject(
            "cycle_profile.retained_labels_exceed_bound",
            "cycle profile exceeds the retained label-character bound",
        )

    backend_graph: nx.Graph[str] = nx.Graph()
    backend_graph.add_nodes_from(graph.vertices)
    backend_graph.add_edges_from(graph.edges)
    cyclic_blocks = [
        component
        for component in biconnected_components(backend_graph)
        if len(component) >= 3
    ]
    # Every simple cycle lies in one biconnected block. Bridges and isolated
    # vertices need no cycle search. Charge decomposition and source-edge scans
    # used to construct every block, in addition to their admitted searches.
    work = vertex_count + len(graph.edges) * (1 + len(cyclic_blocks))
    blocks: list[_BlockPlan] = []
    witness_characters: dict[int, int] = {}
    for component in cyclic_blocks:
        block = SimpleUndirectedGraph(
            vertices=tuple(vertex for vertex in graph.vertices if vertex in component),
            edges=tuple(
                (left, right)
                for left, right in graph.edges
                if left in component and right in component
            ),
        )
        wheel_order = _wheel_search_order(block)
        cycle_order = None
        if len(block.edges) == len(block.vertices):
            # A biconnected simple graph with |E|=|V| is a chordless cycle.
            adjacency: dict[str, list[str]] = {vertex: [] for vertex in block.vertices}
            for left, right in block.edges:
                adjacency[left].append(right)
                adjacency[right].append(left)
            order = [block.vertices[0]]
            previous = None
            while len(order) < len(block.vertices):
                nxt = next(
                    vertex for vertex in adjacency[order[-1]] if vertex != previous
                )
                previous = order[-1]
                order.append(nxt)
            cycle_order = _canonicalize_cycle(tuple(order))
            work += len(block.vertices) ** 2
        else:
            work += _maximum_path_work(block, is_wheel=wheel_order is not None)
        if work > MAX_SEARCH_WORK:
            _reject(
                "cycle_profile.work_bound",
                "complete cycle-profile search exceeds the admitted work bound",
            )
        blocks.append(_BlockPlan(block, wheel_order, cycle_order))
        lengths = (
            (len(block.vertices),) if cycle_order else range(3, len(block.vertices) + 1)
        )
        label_lengths = sorted((len(vertex) for vertex in block.vertices), reverse=True)
        for length in lengths:
            witness_characters[length] = max(
                witness_characters.get(length, 0), sum(label_lengths[:length])
            )
    retained_label_characters += sum(witness_characters.values())
    if retained_label_characters > MAX_CYCLE_PROFILE_RETAINED_LABEL_CHARACTERS:
        _reject(
            "cycle_profile.retained_labels_exceed_bound",
            "cycle profile exceeds the retained label-character bound",
        )
    return _AdmissionPlan(graph=graph, blocks=tuple(blocks))


def compute_cycle_length_profile(
    graph: SimpleUndirectedGraph,
) -> CycleLengthProfileResult:
    """Return the complete cycle-length profile of a simple graph.

    For each length k from 3 to |V|, check if the graph contains a simple
    k-cycle. Return one canonical witness cycle for each present length.
    """
    plan = _admit(graph)
    found: dict[int, tuple[str, ...]] = {}
    candidates: tuple[tuple[int, tuple[str, ...]], ...]
    for block in plan.blocks:
        if block.cycle_order is not None:
            candidates = ((len(block.cycle_order), block.cycle_order),)
        else:
            vertices = list(block.wheel_order or block.graph.vertices)
            n = len(vertices)
            vertex_to_idx = {v: i for i, v in enumerate(vertices)}
            adj_matrix = [[False] * n for _ in range(n)]
            for a, b in block.graph.edges:
                i, j = vertex_to_idx[a], vertex_to_idx[b]
                adj_matrix[i][j] = True
                adj_matrix[j][i] = True
            candidates = tuple(
                (length, witness)
                for length in range(3, n + 1)
                if (witness := _find_cycle_of_length(length, n, adj_matrix, vertices))
                is not None
            )
        for length, witness in candidates:
            if length not in found or witness < found[length]:
                found[length] = witness

    rows = [
        CycleLengthRow._from_kernel(cycle_length=k, witness=w)
        for k, w in sorted(found.items())
    ]
    return CycleLengthProfileResult._from_kernel(graph, tuple(rows))


def verify_cycle_length_profile(claim: CycleLengthProfileResult) -> bool:
    """Return whether a claim has every and only the graph's cycle lengths."""
    try:
        if not all(verify_cycle_length_row(claim.graph, row) for row in claim.rows):
            return False
        return compute_cycle_length_profile(claim.graph).rows == claim.rows
    except (AttributeError, OperationDomainValidationError, TypeError):
        return False


def verify_cycle_length_row(graph: SimpleUndirectedGraph, row: CycleLengthRow) -> bool:
    """Return whether ``row.witness`` is a closed simple cycle in ``graph``."""
    try:
        if row.cycle_length != len(row.witness):
            return False
        if len(set(row.witness)) != len(row.witness):
            return False
        if not set(row.witness) <= set(graph.vertices):
            return False
        edges = {frozenset(edge) for edge in graph.edges}
        cycle = (*row.witness, row.witness[0])
        return all(frozenset((left, right)) in edges for left, right in pairwise(cycle))
    except (AttributeError, TypeError):
        return False


def _find_cycle_of_length(
    length: int,
    n: int,
    adj_matrix: list[list[bool]],
    vertices: list[str],
) -> tuple[str, ...] | None:
    """Find one simple cycle of the given length using DFS backtracking."""

    def dfs(
        start: int,
        current: int,
        visited: list[bool],
        path: list[int],
    ) -> tuple[str, ...] | None:
        if len(path) == length:
            if adj_matrix[current][start]:
                return tuple(vertices[i] for i in path)
            return None
        # The initial vertex is the minimum index in the cycle.  All other
        # vertices are therefore eligible at every depth; restricting them to
        # ``current + 1`` misses cycles whose indices go down and then up.
        for nxt in range(start + 1, n):
            if not visited[nxt] and adj_matrix[current][nxt]:
                visited[nxt] = True
                path.append(nxt)
                result = dfs(start, nxt, visited, path)
                if result is not None:
                    return result
                path.pop()
                visited[nxt] = False
        return None

    for start in range(n):
        visited = [False] * n
        visited[start] = True
        result = dfs(start, start, visited, [start])
        if result is not None:
            canonical = _canonicalize_cycle(result)
            return canonical
    return None


def _canonicalize_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Return the lexicographically smallest rotation in either orientation."""

    return dihedral_canonical_cycle(cycle)


def enumerate_fixed_length_cycles(
    graph: SimpleUndirectedGraph,
    cycle_length: int,
    *,
    chordless: bool = False,
) -> FixedLengthCycleEnumerationResult:
    """Return every simple (or, privately, chordless) cycle of one length."""

    request_checkpoint("before fixed-length cycle enumeration")
    plan = _admit_fixed_cycle_enumeration(graph, cycle_length, chordless=chordless)
    if plan is None:
        request_checkpoint("before empty fixed-length cycle result construction")
        return _empty_cycle_enumeration_result(graph, cycle_length, chordless=chordless)
    adjacency = plan.adjacency
    edge_set = {frozenset(edge) for edge in graph.edges}
    cycles: set[tuple[str, ...]] = set()
    visited_prefixes = 0

    def search_from(start: str) -> None:
        path = [start]
        used = {start}

        def visit(current: str) -> None:
            nonlocal visited_prefixes
            visited_prefixes += 1
            if visited_prefixes % 1024 == 0:
                request_checkpoint("during fixed-length cycle enumeration")
            if len(path) == cycle_length:
                if start not in adjacency[current]:
                    return
                cycle = tuple(path)
                if chordless and any(
                    frozenset((cycle[left], cycle[right])) in edge_set
                    for left in range(cycle_length)
                    for right in range(left + 1, cycle_length)
                    if (right - left) not in (1, cycle_length - 1)
                ):
                    return
                cycles.add(_canonicalize_cycle(cycle))
                return
            for neighbor in adjacency[current]:
                if neighbor <= start or neighbor in used:
                    continue
                used.add(neighbor)
                path.append(neighbor)
                visit(neighbor)
                path.pop()
                used.remove(neighbor)

        visit(start)

    for start in plan.core_vertices:
        search_from(start)
    ordered = tuple(sorted(cycles))
    request_checkpoint("before fixed-length cycle result construction")
    vertex_indices: dict[str, list[int]] = {vertex: [] for vertex in graph.vertices}
    edge_indices: dict[frozenset[str], list[int]] = {
        frozenset(edge): [] for edge in graph.edges
    }
    for index, cycle in enumerate(ordered):
        if index % 1024 == 0:
            request_checkpoint("during fixed-length cycle incidence assembly")
        for vertex in cycle:
            vertex_indices[vertex].append(index)
        for position in range(cycle_length):
            edge_indices[
                frozenset((cycle[position], cycle[(position + 1) % cycle_length]))
            ].append(index)
    vertex_rows = tuple(
        CycleIncidenceRow(source=(vertex,), cycle_indices=tuple(vertex_indices[vertex]))
        for vertex in graph.vertices
    )
    edge_rows = tuple(
        CycleIncidenceRow(
            source=edge,
            cycle_indices=tuple(edge_indices[frozenset(edge)]),
        )
        for edge in graph.edges
    )
    request_checkpoint("after fixed-length cycle incidence assembly")
    return FixedLengthCycleEnumerationResult._from_kernel(
        graph=graph,
        cycle_length=cycle_length,
        family_kind=(
            CycleFamilyKind.CHORDLESS if chordless else CycleFamilyKind.SIMPLE
        ),
        cycles=ordered,
        vertex_incidence=vertex_rows,
        edge_incidence=edge_rows,
    )


def enumerate_chordless_fixed_length_cycles(
    graph: SimpleUndirectedGraph, cycle_length: int
) -> FixedLengthCycleEnumerationResult:
    """Return every induced simple cycle of one fixed length."""

    return enumerate_fixed_length_cycles(graph, cycle_length, chordless=True)


@dataclass(frozen=True, slots=True)
class _FixedCyclePlan:
    adjacency: dict[str, tuple[str, ...]]
    core_vertices: tuple[str, ...]


def _complete_multipartite_part_sizes(
    core_vertices: tuple[str, ...],
    adjacency_sets: dict[str, set[str]],
) -> tuple[int, ...] | None:
    """Return independent-set sizes when the 2-core is complete multipartite."""

    if not core_vertices:
        return ()
    vertex_set = set(core_vertices)
    parts: dict[frozenset[str], set[str]] = {}
    for vertex in core_vertices:
        neighbors = frozenset(adjacency_sets[vertex])
        parts.setdefault(neighbors, set()).add(vertex)
    sizes: list[int] = []
    for neighbors, part in parts.items():
        expected_neighbors = vertex_set - part
        if neighbors != expected_neighbors:
            return None
        if any(adjacency_sets[vertex] != expected_neighbors for vertex in part):
            return None
        sizes.append(len(part))
    return tuple(sizes)


def _chordless_four_cycle_count(part_sizes: tuple[int, ...]) -> int:
    """Count induced 4-cycles in a complete multipartite graph."""

    count = 0
    for left_index, left in enumerate(part_sizes):
        left_pairs = left * (left - 1) // 2
        if left_pairs == 0:
            continue
        for right in part_sizes[left_index + 1 :]:
            count += left_pairs * (right * (right - 1) // 2)
    return count


def _reject_fixed_cycle_resource(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("cycle_length",), code=code, message=message
    )


def _falling_factorial(n: int, k: int) -> int:
    result = 1
    for offset in range(k):
        result *= n - offset
    return result


def _admit_fixed_cycle_enumeration(
    graph: SimpleUndirectedGraph, cycle_length: int, *, chordless: bool = False
) -> _FixedCyclePlan | None:
    """Admit all traversal, intermediate, output, and retained-value quantities."""

    if not isinstance(graph, SimpleUndirectedGraph):
        _reject(
            "cycle_enumeration.graph_type",
            "graph must be a canonical simple undirected graph",
        )
    _validate_graph_carrier(graph)
    if type(cycle_length) is not int or not 3 <= cycle_length <= MAX_VERTICES:
        _reject(
            "cycle_enumeration.length",
            f"cycle_length must be an integer in 3..{MAX_VERTICES}",
        )
    vertex_count = len(graph.vertices)
    if vertex_count > MAX_VERTICES:
        _reject(
            "cycle_enumeration.vertex_bound",
            f"cycle enumeration supports at most {MAX_VERTICES} vertices",
        )

    core_vertices = tuple(sorted(_cycle_core_vertices(graph)))
    core_set = set(core_vertices)
    adjacency_sets: dict[str, set[str]] = {vertex: set() for vertex in core_vertices}
    for left, right in graph.edges:
        # Every simple cycle is contained in the 2-core. Restricting the
        # search carrier here keeps attached trees out of the charged DFS.
        if left in core_set and right in core_set:
            adjacency_sets[left].add(right)
            adjacency_sets[right].add(left)
    adjacency = {
        vertex: tuple(sorted(neighbors)) for vertex, neighbors in adjacency_sets.items()
    }
    source_characters = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    largest_label = max((len(vertex) for vertex in graph.vertices), default=0)
    # A length above the 2-core order is an exact empty result.  This cheap
    # presolve keeps bridge-heavy graphs from inheriting the ambient order.
    if cycle_length > len(core_vertices):
        _admit_fixed_cycle_result(
            graph,
            cycle_length,
            cycle_upper_bound=0,
            source_characters=source_characters,
            largest_label=largest_label,
        )
        return None

    core_order = len(core_vertices)
    max_core_degree = max(
        (len(neighbors) for neighbors in adjacency_sets.values()), default=0
    )
    part_sizes = (
        _complete_multipartite_part_sizes(core_vertices, adjacency_sets)
        if chordless
        else None
    )
    chordless_four_cycle_bound: int | None = None
    # Complete-multipartite 2-cores have no induced cycle of length 5 or more,
    # and their induced 4-cycles are exactly the 2+2 selections from distinct
    # parts. Empty families and the exact C4 count are therefore linear in the
    # core adjacency instead of inheriting a complete-graph traversal bound.
    if chordless and part_sizes is not None:
        if cycle_length >= 5:
            _admit_fixed_cycle_result(
                graph,
                cycle_length,
                cycle_upper_bound=0,
                source_characters=source_characters,
                largest_label=largest_label,
            )
            return None
        if cycle_length == 4:
            chordless_four_cycle_bound = _chordless_four_cycle_count(part_sizes)
            if chordless_four_cycle_bound == 0:
                _admit_fixed_cycle_result(
                    graph,
                    cycle_length,
                    cycle_upper_bound=0,
                    source_characters=source_characters,
                    largest_label=largest_label,
                )
                return None
    # A complete 2-core is a clique: every simple k-cycle with k >= 4 has a
    # chord. Chordless enumeration therefore returns the empty family after a
    # linear core inspection instead of inheriting the all-simple-cycle bound.
    if (
        chordless
        and cycle_length >= 4
        and chordless_four_cycle_bound is None
        and core_order >= 2
        and max_core_degree == core_order - 1
        and all(
            len(neighbors) == core_order - 1 for neighbors in adjacency_sets.values()
        )
    ):
        _admit_fixed_cycle_result(
            graph,
            cycle_length,
            cycle_upper_bound=0,
            source_characters=source_characters,
            largest_label=largest_label,
        )
        return None
    # Complete-graph bounds over the core order. They stay sound for dense
    # graphs but reject cheaply executable sparse instances (for example a
    # bare ring, whose only cycles are found by a linear DFS).
    prefix_bound = core_order
    for depth in range(1, cycle_length):
        prefix_bound += core_order * _falling_factorial(core_order - 1, depth)
    # Each visited prefix scans at most the whole core adjacency tuple. A
    # terminal prefix also pays for canonicalization and, for the chordless
    # operation, every unordered vertex pair.
    scan_bound = prefix_bound * max(1, core_order)
    terminal_bound = _falling_factorial(core_order, cycle_length)
    complete_cycle_upper_bound = terminal_bound // (2 * cycle_length)
    terminal_checks = 1 + 2 * cycle_length
    if chordless:
        terminal_checks += cycle_length * (cycle_length - 1) // 2
    complete_work = (
        scan_bound
        + terminal_bound * terminal_checks
        + 3 * (vertex_count + len(graph.edges))
    )
    # Topology-sensitive bounds from the built core adjacency. From each
    # start, length-d simple paths number at most Δ·(Δ-1)^(d-1): the first
    # step has at most Δ choices and every later step revisits the
    # predecessor, leaving at most Δ-1 unvisited neighbors. Each found cycle
    # needs at least one length-(k-1) terminal visit, so those visits also
    # bound the output. Taking the minimum with the complete-graph bounds
    # keeps dense-graph behavior unchanged while admitting sparse graphs.
    branching = max(max_core_degree - 1, 0)
    directed_paths = core_order * max_core_degree
    topology_prefix_total = core_order + directed_paths
    for _depth in range(2, cycle_length):
        directed_paths *= branching
        topology_prefix_total += directed_paths
    topology_work = (
        topology_prefix_total * max(max_core_degree, 1)
        + directed_paths * terminal_checks
        + 3 * (vertex_count + len(graph.edges))
    )
    cycle_upper_bound = min(complete_cycle_upper_bound, directed_paths)
    if chordless_four_cycle_bound is not None:
        cycle_upper_bound = chordless_four_cycle_bound
    if min(complete_work, topology_work) > MAX_FIXED_CYCLE_WORK:
        _reject_fixed_cycle_resource(
            "cycle_enumeration.work_bound",
            "complete fixed-length traversal and incidence assembly exceed the admitted work envelope",
        )
    _admit_fixed_cycle_result(
        graph,
        cycle_length,
        cycle_upper_bound=cycle_upper_bound,
        source_characters=source_characters,
        largest_label=largest_label,
    )
    return _FixedCyclePlan(adjacency=adjacency, core_vertices=core_vertices)


def _validate_graph_carrier(graph: SimpleUndirectedGraph) -> None:
    """Keep forged direct-native graph carriers on the typed error path."""

    vertices = graph.vertices
    if type(vertices) is not tuple or any(
        type(vertex) is not str for vertex in vertices
    ):
        _reject(
            "cycle_enumeration.graph_structure",
            "graph vertices must be a tuple of string labels",
        )
    for vertex in vertices:
        try:
            vertex.encode("utf-8")
        except UnicodeEncodeError:
            _reject(
                "cycle_enumeration.graph_structure",
                "graph vertices must contain only valid Unicode scalar values",
            )
        if not unicodedata.is_normalized("NFC", vertex):
            _reject(
                "cycle_enumeration.graph_structure",
                "graph vertices must use Unicode NFC so results round-trip",
            )
    vertex_set = set(vertices)
    if len(vertex_set) != len(vertices):
        _reject(
            "cycle_enumeration.graph_structure",
            "graph vertices must be unique",
        )
    edges = graph.edges
    if type(edges) is not tuple:
        _reject(
            "cycle_enumeration.graph_structure",
            "graph edges must be a tuple of canonical pairs",
        )
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if (
            type(edge) is not tuple
            or len(edge) != 2
            or any(type(endpoint) is not str for endpoint in edge)
            or edge[0] >= edge[1]
            or edge[0] not in vertex_set
            or edge[1] not in vertex_set
            or edge in seen
        ):
            _reject(
                "cycle_enumeration.graph_structure",
                "graph edges must be unique canonical pairs of declared vertices",
            )
        seen.add(edge)


def _admit_fixed_cycle_result(
    graph: SimpleUndirectedGraph,
    cycle_length: int,
    *,
    cycle_upper_bound: int,
    source_characters: int,
    largest_label: int,
) -> None:
    if cycle_upper_bound > MAX_FIXED_CYCLES:
        _reject_fixed_cycle_resource(
            "cycle_enumeration.output_bound",
            "the complete fixed-length cycle family exceeds the admitted result envelope",
        )
    # Reserve source axes, cycle labels, incidence indexes, and index
    # materialization before the first DFS branch.
    incidence_rows = len(graph.vertices) + len(graph.edges)
    index_digits = len(str(max(1, cycle_upper_bound)))
    retained_characters = (
        source_characters * 2
        + cycle_upper_bound * cycle_length * largest_label
        + incidence_rows * (2 * largest_label + index_digits + 32)
        + cycle_upper_bound * cycle_length * index_digits * 2
    )
    if retained_characters > MAX_FIXED_CYCLE_RETAINED_LABEL_CHARACTERS:
        _reject_fixed_cycle_resource(
            "cycle_enumeration.retained_labels_exceed_bound",
            "the complete fixed-length cycle result exceeds the admitted retained-label envelope",
        )


def _empty_cycle_enumeration_result(
    graph: SimpleUndirectedGraph, cycle_length: int, *, chordless: bool
) -> FixedLengthCycleEnumerationResult:
    """Construct an admitted empty family with all source axes retained."""

    vertex_incidence: list[CycleIncidenceRow] = []
    for index, vertex in enumerate(graph.vertices):
        if index % 64 == 0:
            request_checkpoint("during empty fixed-length cycle incidence assembly")
        vertex_incidence.append(CycleIncidenceRow(source=(vertex,), cycle_indices=()))
    edge_incidence: list[CycleIncidenceRow] = []
    for index, edge in enumerate(graph.edges):
        if index % 64 == 0:
            request_checkpoint("during empty fixed-length cycle incidence assembly")
        edge_incidence.append(CycleIncidenceRow(source=edge, cycle_indices=()))
    request_checkpoint("after empty fixed-length cycle incidence assembly")
    return FixedLengthCycleEnumerationResult._from_kernel(
        graph=graph,
        cycle_length=cycle_length,
        family_kind=(
            CycleFamilyKind.CHORDLESS if chordless else CycleFamilyKind.SIMPLE
        ),
        cycles=(),
        vertex_incidence=tuple(vertex_incidence),
        edge_incidence=tuple(edge_incidence),
    )
