"""Cycle-length profile kernel."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from itertools import combinations, pairwise

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


def _wheel_cycle_count(wheel_order: tuple[str, ...], cycle_length: int) -> int:
    """Exact number of simple or chordless `cycle_length`-cycles in a wheel.

    ``wheel_order`` is ``(hub, rim...)`` in cyclic rim order. Simple cycles are
    the ``m`` hub-plus-consecutive-arc families for each arc length plus the rim
    cycle; a wheel has no induced cycle longer than four, so the chordless
    family is separately counted.
    """

    rim_count = len(wheel_order) - 1
    if cycle_length < 3 or cycle_length > rim_count + 1:
        return 0
    if cycle_length == rim_count + 1:
        # A Hamiltonian cycle uses the whole rim plus the hub; removing any one
        # of the ``rim_count`` rim edges gives a distinct Hamiltonian cycle.
        return rim_count
    # Hub cycles: one per consecutive arc of ``cycle_length - 1`` rim vertices,
    # of which there are ``rim_count`` proper arcs.
    count = rim_count
    if cycle_length == rim_count:
        # The rim itself is an additional cycle that avoids the hub.
        count += 1
    return count


def _wheel_chordless_cycle_count(
    wheel_order: tuple[str, ...], cycle_length: int
) -> int:
    """Exact number of induced `cycle_length`-cycles in a wheel."""

    rim_count = len(wheel_order) - 1
    if cycle_length == rim_count:
        # The rim itself is an induced cycle (a wheel rim has no chords) for
        # length at least five; a four-vertex rim is also induced.
        return 1 if rim_count >= 4 else 0
    if cycle_length == 3:
        # Triangles use the hub plus a rim edge, one per rim edge; the rim edge
        # is a chord-free triangle.
        return rim_count
    if cycle_length == 4:
        # Hub + three consecutive rim vertices has the chord hub-to-middle, so
        # there is no induced four-cycle through the hub.
        return 0
    return 0


def _wheel_order_for_block(
    block: tuple[str, ...], local: dict[str, set[str]]
) -> tuple[str, ...] | None:
    """Return ``(hub, rim...)`` when the block is a wheel, else ``None``."""

    vertex_count = len(block)
    if vertex_count < 4:
        return None
    hubs = [vertex for vertex in block if len(local[vertex]) == vertex_count - 1]
    if len(hubs) != 1:
        return None
    hub = hubs[0]
    rim = [vertex for vertex in block if vertex != hub]
    if any(len(local[vertex]) != 3 for vertex in rim):
        return None
    rim_adjacency = {vertex: local[vertex] - {hub} for vertex in rim}
    if any(len(neighbors) != 2 for neighbors in rim_adjacency.values()):
        return None
    start = sorted(rim)[0]
    order = [start]
    previous: str | None = None
    current = start
    while len(order) < len(rim):
        candidates = sorted(
            neighbor for neighbor in rim_adjacency[current] if neighbor != previous
        )
        next_vertex = next(
            candidate for candidate in candidates if candidate not in order
        )
        if next_vertex is None:
            return None
        order.append(next_vertex)
        previous, current = current, next_vertex
    return (hub, *order)


def _fan_order_for_block(
    block: tuple[str, ...], local: dict[str, set[str]]
) -> tuple[str, ...] | None:
    """Return ``(hub, path...)`` when the block is a fan, else ``None``.

    A fan joins one hub to every vertex of an ``m``-vertex path (``m >= 2``),
    giving ``m + 1`` vertices and ``2m - 1`` edges.
    """

    vertex_count = len(block)
    if vertex_count < 3:
        return None
    hubs = [vertex for vertex in block if len(local[vertex]) == vertex_count - 1]
    if len(hubs) != 1:
        return None
    hub = hubs[0]
    path = [vertex for vertex in block if vertex != hub]
    path_adjacency = {vertex: local[vertex] - {hub} for vertex in path}
    endpoints = sorted(vertex for vertex in path if len(path_adjacency[vertex]) == 1)
    if len(endpoints) != 2 or any(
        len(neighbors) not in (1, 2) for neighbors in path_adjacency.values()
    ):
        return None
    order = [endpoints[0]]
    previous: str | None = None
    current = endpoints[0]
    while len(order) < len(path):
        candidates = sorted(
            neighbor for neighbor in path_adjacency[current] if neighbor != previous
        )
        next_vertex = next(
            (candidate for candidate in candidates if candidate not in order), None
        )
        if next_vertex is None:
            return None
        order.append(next_vertex)
        previous, current = current, next_vertex
    return (hub, *order)


def _fan_cycle_count(fan_order: tuple[str, ...], cycle_length: int) -> int:
    """Exact simple-cycle count of a fan block.

    Cycles use the hub plus a contiguous path segment of ``cycle_length - 1``
    path vertices, of which a path of ``m`` vertices has ``m - L + 2``.
    """

    path_count = len(fan_order) - 1
    arc_length = cycle_length - 1
    if arc_length < 2 or arc_length > path_count:
        return 0
    return path_count - arc_length + 1


def _fan_chordless_cycle_count(fan_order: tuple[str, ...], cycle_length: int) -> int:
    """Exact induced-cycle count of a fan block.

    Only a triangle (hub plus one path edge) is induced; any longer hub cycle
    has a chord from the hub to an interior path vertex, and the path has no
    cycle of its own.
    """

    if cycle_length != 3:
        return 0
    return max(0, len(fan_order) - 2)


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


def _fan_cycles(
    fan_order: tuple[str, ...], cycle_length: int, *, chordless: bool
) -> set[tuple[str, ...]]:
    """Enumerate the exact cycles of one recognized fan block.

    ``fan_order`` is ``(hub, path...)``. Every cycle uses the hub plus a
    contiguous path segment; only the triangle is induced.
    """

    hub = fan_order[0]
    path = fan_order[1:]
    path_count = len(path)
    cycles: set[tuple[str, ...]] = set()
    arc_length = cycle_length - 1
    if chordless:
        if cycle_length == 3:
            for index in range(path_count - 1):
                cycles.add(_canonicalize_cycle((hub, path[index], path[index + 1])))
        return cycles
    if arc_length < 2 or arc_length > path_count:
        return cycles
    for start in range(path_count - arc_length + 1):
        segment = tuple(path[start : start + arc_length])
        cycles.add(_canonicalize_cycle((hub, *segment)))
    return cycles


def _wheel_cycles(
    wheel_order: tuple[str, ...], cycle_length: int, *, chordless: bool
) -> set[tuple[str, ...]]:
    """Enumerate the exact cycles of one recognized wheel block.

    ``wheel_order`` is ``(hub, rim...)`` in cyclic rim order. Emitting the
    family directly avoids a DFS that explores every hub placement between rim
    arcs, which the admission work bound cannot charge tightly.
    """

    hub = wheel_order[0]
    rim = wheel_order[1:]
    rim_count = len(rim)
    cycles: set[tuple[str, ...]] = set()
    if chordless:
        if cycle_length == 3:
            for index in range(rim_count):
                cycles.add(
                    _canonicalize_cycle((hub, rim[index], rim[(index + 1) % rim_count]))
                )
        elif cycle_length == rim_count and rim_count >= 4:
            cycles.add(_canonicalize_cycle(rim))
        return cycles
    if cycle_length == rim_count + 1:
        # Hamiltonians: the hub plus all ``rim_count`` rim vertices, which is a
        # rim path between two rim vertices; each removed rim edge gives one.
        for start in range(rim_count):
            arc = tuple(
                rim[(start + offset) % rim_count] for offset in range(rim_count)
            )
            cycles.add(_canonicalize_cycle((hub, *arc)))
        return cycles
    if cycle_length < 3 or cycle_length > rim_count:
        return cycles
    arc_length = cycle_length - 1
    for start in range(rim_count):
        arc = tuple(rim[(start + offset) % rim_count] for offset in range(arc_length))
        cycles.add(_canonicalize_cycle((hub, *arc)))
    if cycle_length == rim_count:
        cycles.add(_canonicalize_cycle(rim))
    return cycles


def _canonicalize_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Return the lexicographically smallest rotation in either orientation."""

    return dihedral_canonical_cycle(cycle)


def enumerate_fixed_length_cycles(
    graph: SimpleUndirectedGraph,
    cycle_length: int,
) -> FixedLengthCycleEnumerationResult:
    """Return every simple cycle of one fixed length."""

    return _enumerate_cycles(graph, cycle_length, chordless=False)


def _enumerate_cycles(
    graph: SimpleUndirectedGraph,
    cycle_length: int,
    *,
    chordless: bool,
) -> FixedLengthCycleEnumerationResult:
    """Return every simple or chordless cycle of one length."""

    request_checkpoint("before fixed-length cycle enumeration")
    plan = _admit_fixed_cycle_enumeration(graph, cycle_length, chordless=chordless)
    if plan is None:
        request_checkpoint("before empty fixed-length cycle result construction")
        return _empty_cycle_enumeration_result(graph, cycle_length, chordless=chordless)
    edge_set = {frozenset(edge) for edge in graph.edges}
    cycles: set[tuple[str, ...]] = set()
    visited_prefixes = 0

    for block in plan.blocks:
        if block.wheel_order is not None:
            cycles.update(
                _wheel_cycles(block.wheel_order, cycle_length, chordless=chordless)
            )
            continue
        if block.fan_order is not None:
            cycles.update(
                _fan_cycles(block.fan_order, cycle_length, chordless=chordless)
            )
            continue
        adjacency = block.adjacency

        def search_from(
            start: str, block_adjacency: dict[str, tuple[str, ...]]
        ) -> None:
            path = [start]
            used = {start}

            def visit(current: str) -> None:
                nonlocal visited_prefixes
                visited_prefixes += 1
                if visited_prefixes % 1024 == 0:
                    request_checkpoint("during fixed-length cycle enumeration")
                if len(path) == cycle_length:
                    if start not in block_adjacency[current]:
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
                for neighbor in block_adjacency[current]:
                    if neighbor <= start or neighbor in used:
                        continue
                    used.add(neighbor)
                    path.append(neighbor)
                    visit(neighbor)
                    path.pop()
                    used.remove(neighbor)

            visit(start)

        for start in block.core_vertices:
            search_from(start, adjacency)
    ordered = tuple(sorted(cycles))
    return _cycle_enumeration_result(graph, cycle_length, ordered, chordless=chordless)


def _cycle_enumeration_result(
    graph: SimpleUndirectedGraph,
    cycle_length: int,
    ordered: tuple[tuple[str, ...], ...],
    *,
    chordless: bool,
) -> FixedLengthCycleEnumerationResult:
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

    return _enumerate_cycles(graph, cycle_length, chordless=True)


@dataclass(frozen=True, slots=True)
class _FixedCycleBlock:
    adjacency: dict[str, tuple[str, ...]]
    core_vertices: tuple[str, ...]
    wheel_order: tuple[str, ...] | None = None
    fan_order: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _FixedCyclePlan:
    blocks: tuple[_FixedCycleBlock, ...]


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


def _multipartite_cycle_exists(part_sizes: tuple[int, ...], cycle_length: int) -> bool:
    """Return whether a complete multipartite graph contains a cycle of this length.

    Edges join distinct parts. A cycle of length ``L`` needs at least three
    parts, or exactly two parts when ``L`` is even and each part supplies
    ``L / 2`` distinct vertices. In every case the cycle cannot exceed the
    total vertex count.
    """

    total = sum(part_sizes)
    if cycle_length > total or cycle_length < 3:
        return False
    if len(part_sizes) < 3:
        # A bipartite graph (exactly two parts, or one) has only even cycles,
        # and a ``cycle_length``-cycle needs ``cycle_length / 2`` vertices from
        # each part.
        return cycle_length % 2 == 0 and 2 * min(part_sizes, default=0) >= cycle_length
    # At least three parts. A cycle alternates between parts, so a part of size
    # ``m`` can contribute at most ``total - m + 1`` of its vertices (each
    # separated by at least one non-part vertex). When the largest part
    # dominates, that caps the longest cycle at ``2 * (total - m)``.
    largest = max(part_sizes)
    longest = total if 2 * largest <= total else 2 * (total - largest)
    return cycle_length <= longest


def _simple_triangle_count(part_sizes: tuple[int, ...]) -> int:
    """Count triangles in a complete multipartite graph.

    A triangle needs one vertex from each of three distinct parts, and every
    such choice is an induced (chordless) triangle.
    """

    count = 0
    for indices in combinations(range(len(part_sizes)), 3):
        count += (
            part_sizes[indices[0]] * part_sizes[indices[1]] * part_sizes[indices[2]]
        )
    return count


def _simple_four_cycle_count(part_sizes: tuple[int, ...]) -> int:
    """Count simple four-cycles in a complete multipartite graph.

    A four-cycle either uses two vertices from each of two parts, two vertices
    from one part and one from each of two others (with a chord between the
    single-part vertices), or one vertex from each of four distinct parts
    (which spans K4 and therefore supports three four-cycles).
    """

    count = _chordless_four_cycle_count(part_sizes)
    for doubled in range(len(part_sizes)):
        pairs = part_sizes[doubled] * (part_sizes[doubled] - 1) // 2
        if pairs == 0:
            continue
        others = [
            part_sizes[index] for index in range(len(part_sizes)) if index != doubled
        ]
        for left_index, left in enumerate(others):
            for right in others[left_index + 1 :]:
                count += pairs * left * right
    # Four distinct parts: the induced four vertices form K4 with three cycles.
    for indices in combinations(range(len(part_sizes)), 4):
        product = 1
        for index in indices:
            product *= part_sizes[index]
        count += 3 * product
    return count


def _chordless_four_cycle_count(part_sizes: tuple[int, ...]) -> int:
    """Count induced 4-cycles in a complete multipartite graph.

    An induced four-cycle uses exactly two vertices from each of two parts; four
    distinct parts would induce K4, which has no induced four-cycle.
    """

    count = 0
    for left_index, left in enumerate(part_sizes):
        left_pairs = left * (left - 1) // 2
        if left_pairs == 0:
            continue
        for right in part_sizes[left_index + 1 :]:
            count += left_pairs * (right * (right - 1) // 2)
    return count


def _core_cyclic_blocks(
    core_vertices: tuple[str, ...],
    adjacency_sets: dict[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    """Return 2-core biconnected blocks that can contain a simple cycle."""

    backend: nx.Graph[str] = nx.Graph()
    backend.add_nodes_from(core_vertices)
    backend.add_edges_from(
        (vertex, neighbor)
        for vertex, neighbors in adjacency_sets.items()
        for neighbor in neighbors
        if vertex < neighbor
    )
    return tuple(
        tuple(sorted(block))
        for block in biconnected_components(backend)
        if len(block) >= 3
    )


def _block_adjacency(
    block: tuple[str, ...], adjacency_sets: dict[str, set[str]]
) -> dict[str, set[str]]:
    block_set = set(block)
    return {vertex: adjacency_sets[vertex] & block_set for vertex in block}


def _chordless_multipartite_block_bound(
    block: tuple[str, ...],
    adjacency_sets: dict[str, set[str]],
    cycle_length: int,
) -> int | None:
    """Exact induced-cycle count for one complete multipartite block.

    Returns the exact count when the block is complete multipartite (zero when
    the block admits no induced cycle of this length) and ``None`` when the
    block is not complete multipartite and needs the generic bound.
    """

    part_sizes = _complete_multipartite_part_sizes(
        block, _block_adjacency(block, adjacency_sets)
    )
    if part_sizes is None:
        return None
    if not _multipartite_cycle_exists(part_sizes, cycle_length):
        return 0
    if cycle_length == 4:
        return _chordless_four_cycle_count(part_sizes)
    # A complete multipartite graph has no induced cycle of length at least
    # five: any longer cycle closes a chord between distinct parts.
    return 0


def _reject_fixed_cycle_resource(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("cycle_length",), code=code, message=message
    )


def _falling_factorial(n: int, k: int) -> int:
    result = 1
    for offset in range(k):
        result *= n - offset
    return result


def _multipartite_triangle_work(
    block: tuple[str, ...], adjacency_sets: dict[str, set[str]]
) -> tuple[int, dict[str, tuple[str, ...]]]:
    """Traversal work and adjacency for a multipartite triangle block."""

    local = _block_adjacency(block, adjacency_sets)
    adjacency = {vertex: tuple(sorted(local[vertex])) for vertex in block}
    return len(block) ** 2, adjacency


def _multipartite_four_cycle_work(
    block: tuple[str, ...], adjacency_sets: dict[str, set[str]]
) -> tuple[int, dict[str, tuple[str, ...]]]:
    """Traversal work and adjacency for a multipartite four-cycle block."""

    local = _block_adjacency(block, adjacency_sets)
    adjacency = {vertex: tuple(sorted(local[vertex])) for vertex in block}
    return len(block) ** 2, adjacency


def _block_fixed_cycle_bounds(
    block: tuple[str, ...],
    adjacency_sets: dict[str, set[str]],
    cycle_length: int,
    *,
    chordless: bool,
) -> tuple[int, int, dict[str, tuple[str, ...]]]:
    """Return traversal work, cycle-count bound, and adjacency for one block."""

    local = _block_adjacency(block, adjacency_sets)
    core_order = len(block)
    adjacency = {vertex: tuple(sorted(local[vertex])) for vertex in block}
    if cycle_length > core_order:
        return 0, 0, adjacency
    wheel_order = _wheel_order_for_block(block, local)
    if wheel_order is not None:
        count = (
            _wheel_chordless_cycle_count(wheel_order, cycle_length)
            if chordless
            else _wheel_cycle_count(wheel_order, cycle_length)
        )
        # The wheel family is enumerated directly, so charge linear work for
        # emitting its arcs rather than the DFS envelope.
        return core_order, count, adjacency
    fan_order = _fan_order_for_block(block, local)
    if fan_order is not None:
        count = (
            _fan_chordless_cycle_count(fan_order, cycle_length)
            if chordless
            else _fan_cycle_count(fan_order, cycle_length)
        )
        return core_order, count, adjacency
    max_core_degree = max((len(local[vertex]) for vertex in block), default=0)
    prefix_bound = core_order
    for depth in range(1, cycle_length):
        prefix_bound += core_order * _falling_factorial(core_order - 1, depth)
    scan_bound = prefix_bound * max(1, core_order)
    terminal_bound = _falling_factorial(core_order, cycle_length)
    complete_cycle_upper_bound = terminal_bound // (2 * cycle_length)
    terminal_checks = 1 + 2 * cycle_length
    if chordless:
        terminal_checks += cycle_length * (cycle_length - 1) // 2
    complete_work = scan_bound + terminal_bound * terminal_checks
    branching = max(max_core_degree - 1, 0)
    directed_paths = core_order * max_core_degree
    topology_prefix_total = core_order + directed_paths
    for _depth in range(2, cycle_length):
        directed_paths *= branching
        topology_prefix_total += directed_paths
    topology_work = (
        topology_prefix_total * max(max_core_degree, 1)
        + directed_paths * terminal_checks
    )
    cycle_upper_bound = min(complete_cycle_upper_bound, directed_paths)
    return min(complete_work, topology_work), cycle_upper_bound, adjacency


def _core_adjacency_sets(
    graph: SimpleUndirectedGraph,
) -> tuple[tuple[str, ...], dict[str, set[str]]]:
    core_vertices = tuple(sorted(_cycle_core_vertices(graph)))
    core_set = set(core_vertices)
    adjacency_sets: dict[str, set[str]] = {vertex: set() for vertex in core_vertices}
    for left, right in graph.edges:
        if left in core_set and right in core_set:
            adjacency_sets[left].add(right)
            adjacency_sets[right].add(left)
    return core_vertices, adjacency_sets


def _admit_empty_fixed_cycle(
    graph: SimpleUndirectedGraph,
    cycle_length: int,
    *,
    source_characters: int,
    largest_label: int,
) -> None:
    _admit_fixed_cycle_result(
        graph,
        cycle_length,
        cycle_upper_bound=0,
        source_characters=source_characters,
        largest_label=largest_label,
    )


def _admit_fixed_cycle_enumeration(
    graph: SimpleUndirectedGraph, cycle_length: int, *, chordless: bool = False
) -> _FixedCyclePlan | None:
    """Admit all traversal, intermediate, output, and retained-value quantities."""

    if not isinstance(graph, SimpleUndirectedGraph):
        _reject(
            "cycle_enumeration.graph_type",
            "graph must be a canonical simple undirected graph",
        )
    vertices = graph.vertices
    if type(vertices) is not tuple:
        _reject(
            "cycle_enumeration.graph_structure",
            "graph vertices must be a tuple of string labels",
        )
    vertex_count = len(vertices)
    if vertex_count > MAX_VERTICES:
        _reject(
            "cycle_enumeration.vertex_bound",
            f"cycle enumeration supports at most {MAX_VERTICES} vertices",
        )
    _validate_graph_carrier(graph)
    if type(cycle_length) is not int or not 3 <= cycle_length <= MAX_VERTICES:
        _reject(
            "cycle_enumeration.length",
            f"cycle_length must be an integer in 3..{MAX_VERTICES}",
        )
    vertex_count = len(graph.vertices)
    core_vertices, adjacency_sets = _core_adjacency_sets(graph)
    source_characters = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    largest_label = max((len(vertex) for vertex in graph.vertices), default=0)
    if cycle_length > len(core_vertices):
        _admit_empty_fixed_cycle(
            graph,
            cycle_length,
            source_characters=source_characters,
            largest_label=largest_label,
        )
        return None
    return _admit_fixed_cycle_search_plan(
        graph,
        cycle_length,
        chordless=chordless,
        vertex_count=vertex_count,
        core_vertices=core_vertices,
        adjacency_sets=adjacency_sets,
        source_characters=source_characters,
        largest_label=largest_label,
    )


def _admit_fixed_cycle_search_plan(
    graph: SimpleUndirectedGraph,
    cycle_length: int,
    *,
    chordless: bool,
    vertex_count: int,
    core_vertices: tuple[str, ...],
    adjacency_sets: dict[str, set[str]],
    source_characters: int,
    largest_label: int,
) -> _FixedCyclePlan | None:
    cyclic_blocks = _core_cyclic_blocks(core_vertices, adjacency_sets)
    core_order = len(core_vertices)
    max_core_degree = max(
        (len(neighbors) for neighbors in adjacency_sets.values()), default=0
    )
    if (
        chordless
        and cycle_length >= 4
        and core_order >= 2
        and max_core_degree == core_order - 1
        and all(
            len(neighbors) == core_order - 1 for neighbors in adjacency_sets.values()
        )
    ):
        # The whole core is a complete graph, which has no induced cycle of
        # length at least four.
        _admit_empty_fixed_cycle(
            graph,
            cycle_length,
            source_characters=source_characters,
            largest_label=largest_label,
        )
        return None
    source_scan = 3 * (vertex_count + len(graph.edges))
    complete_work = source_scan
    cycle_upper_bound = 0
    search_blocks: list[_FixedCycleBlock] = []
    for block in cyclic_blocks:
        exact_chordless = None
        if chordless and cycle_length >= 4:
            exact_chordless = _chordless_multipartite_block_bound(
                block, adjacency_sets, cycle_length
            )
        if exact_chordless is not None:
            cycle_upper_bound += exact_chordless
            if exact_chordless and cycle_length == 4:
                # The block is enumerated by DFS; charge its traversal work even
                # though the exact induced-cycle count is already known.
                block_work, _, block_adjacency = _block_fixed_cycle_bounds(
                    block, adjacency_sets, cycle_length, chordless=chordless
                )
                complete_work += block_work
                search_blocks.append(
                    _FixedCycleBlock(adjacency=block_adjacency, core_vertices=block)
                )
            continue
        part_sizes = _complete_multipartite_part_sizes(
            block, _block_adjacency(block, adjacency_sets)
        )
        if part_sizes is not None and not _multipartite_cycle_exists(
            part_sizes, cycle_length
        ):
            # A recognized multipartite block with no cycle of this length
            # contributes nothing and must not pay the generic all-vertex bound.
            continue
        if part_sizes is not None and cycle_length == 3:
            # Every triangle selects one vertex from each of three distinct
            # parts, so its exact count is a bounded polynomial in the part
            # sizes rather than the generic all-vertex permutation bound.
            block_work, block_adjacency = _multipartite_triangle_work(
                block, adjacency_sets
            )
            complete_work += block_work
            cycle_upper_bound += _simple_triangle_count(part_sizes)
            search_blocks.append(
                _FixedCycleBlock(adjacency=block_adjacency, core_vertices=block)
            )
            continue
        if part_sizes is not None and cycle_length == 4 and not chordless:
            # The simple four-cycles of a complete multipartite graph are the
            # induced family plus the three-part cycles that take two vertices
            # from one part and one from each of two others (which have a
            # chord), so use the exact simple count instead of the
            # complete-graph bound.
            block_work, block_adjacency = _multipartite_four_cycle_work(
                block, adjacency_sets
            )
            complete_work += block_work
            cycle_upper_bound += _simple_four_cycle_count(part_sizes)
            search_blocks.append(
                _FixedCycleBlock(adjacency=block_adjacency, core_vertices=block)
            )
            continue
        block_work, block_cycles, block_adjacency = _block_fixed_cycle_bounds(
            block, adjacency_sets, cycle_length, chordless=chordless
        )
        complete_work += block_work
        cycle_upper_bound += block_cycles
        if block_cycles:
            search_blocks.append(
                _FixedCycleBlock(
                    adjacency=block_adjacency,
                    core_vertices=block,
                    wheel_order=_wheel_order_for_block(
                        block, _block_adjacency(block, adjacency_sets)
                    ),
                    fan_order=_fan_order_for_block(
                        block, _block_adjacency(block, adjacency_sets)
                    ),
                )
            )
    if complete_work > MAX_FIXED_CYCLE_WORK:
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
    if not search_blocks or cycle_upper_bound == 0:
        return None
    return _FixedCyclePlan(blocks=tuple(search_blocks))


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
    source_characters = 0
    for vertex in vertices:
        source_characters += len(vertex)
        if source_characters > MAX_FIXED_CYCLE_RETAINED_LABEL_CHARACTERS // 2:
            # ``len`` is O(1), while encoding and NFC-scanning below walk the
            # whole label. The retained envelope always reserves at least twice
            # the source characters, so exceeding half the ceiling here means
            # the later envelope check would refuse the input anyway; reject
            # before the full-string operations.
            _reject_fixed_cycle_resource(
                "cycle_enumeration.retained_labels_exceed_bound",
                "the complete fixed-length cycle result exceeds the admitted retained-label envelope",
            )
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
