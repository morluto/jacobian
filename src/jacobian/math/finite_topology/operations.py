"""Domain-owned finite topology kernels."""

from __future__ import annotations

from collections import deque

from jacobian.math.finite_topology.values import FiniteTopology, PointMap

__all__ = [
    "beat_points",
    "closure",
    "connected_components",
    "interior",
    "is_continuous",
    "minimal_open_neighborhoods",
    "specialization_preorder",
]


def _open_set_set(topology: FiniteTopology) -> set[frozenset[int]]:
    return {frozenset(op) for op in topology.open_sets}


def specialization_preorder(
    topology: FiniteTopology,
) -> tuple[tuple[bool, ...], ...]:
    """Return the specialization preorder as a boolean matrix.

    ``preorder[i][j]`` is True iff ``j`` is in the closure of ``{i}``,
    i.e., every open set containing ``i`` also contains ``j``.
    """
    n = topology.point_count
    opens_by_point: list[list[frozenset[int]]] = [[] for _ in range(n)]
    for op in topology.open_sets:
        fs = frozenset(op)
        for pt in fs:
            opens_by_point[pt].append(fs)
    matrix: list[tuple[bool, ...]] = []
    for i in range(n):
        row: list[bool] = []
        for j in range(n):
            row.append(all(j in op for op in opens_by_point[i]))
        matrix.append(tuple(row))
    return tuple(matrix)


def minimal_open_neighborhoods(
    topology: FiniteTopology,
) -> tuple[frozenset[int], ...]:
    """Return the minimal open neighborhood U_x for each point x."""
    n = topology.point_count
    result: list[frozenset[int]] = []
    for pt in range(n):
        containing = [frozenset(op) for op in topology.open_sets if pt in op]
        if not containing:
            result.append(frozenset())
        else:
            result.append(min(containing, key=len))
    return tuple(result)


def closure(
    topology: FiniteTopology,
    subset: tuple[int, ...] | frozenset[int],
) -> frozenset[int]:
    """Compute the closure of a subset.

    The closure is the smallest closed set containing the subset.
    A closed set is the complement of an open set.
    """
    opens = _open_set_set(topology)
    n = topology.point_count
    full = set(range(n))
    sub = set(subset)
    # closure = complement of the union of all open sets disjoint from sub
    disjoint_union: set[int] = set()
    for op in opens:
        if op.isdisjoint(sub):
            disjoint_union |= op
    return frozenset(full - disjoint_union)


def interior(
    topology: FiniteTopology,
    subset: tuple[int, ...] | frozenset[int],
) -> frozenset[int]:
    """Compute the interior of a subset (largest open set contained in the subset)."""
    sub = set(subset)
    opens = _open_set_set(topology)
    result: set[int] = set()
    for op in opens:
        if op <= sub:
            result |= op
    return frozenset(result)


def connected_components(
    topology: FiniteTopology,
) -> tuple[tuple[int, ...], ...]:
    """Compute connected components via the specialization preorder graph.

    Two points are connected if there's a path using both directions of the
    preorder relation.
    """
    n = topology.point_count
    preorder = specialization_preorder(topology)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if preorder[i][j] or preorder[j][i]:
                adj[i].add(j)
                adj[j].add(i)
    visited: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(n):
        if start in visited:
            continue
        queue: deque[int] = deque([start])
        visited.add(start)
        component: list[int] = [start]
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def is_continuous(
    domain: FiniteTopology,
    codomain: FiniteTopology,
    function: PointMap,
) -> bool:
    """Check if a point map is continuous.

    A map f: X -> Y is continuous iff the preimage of every open set in Y
    is open in X.
    """
    opens_x = _open_set_set(domain)
    opens_y = _open_set_set(codomain)
    for open_y in opens_y:
        preimage: set[int] = set()
        for i, target in enumerate(function.function):
            if target in open_y:
                preimage.add(i)
        if frozenset(preimage) not in opens_x:
            return False
    return True


def _maximal_elements(
    below: set[int], preorder: tuple[tuple[bool, ...], ...],
) -> list[int]:
    """Return maximal elements of ``below`` under the preorder."""
    maximals: list[int] = []
    for m in below:
        is_max = True
        for other in below:
            if other != m and preorder[m][other]:
                is_max = False
                break
        if is_max:
            maximals.append(m)
    return maximals


def _minimal_elements(
    above: set[int], preorder: tuple[tuple[bool, ...], ...],
) -> list[int]:
    """Return minimal elements of ``above`` under the preorder."""
    minimals: list[int] = []
    for m in above:
        is_min = True
        for other in above:
            if other != m and preorder[other][m]:
                is_min = False
                break
        if is_min:
            minimals.append(m)
    return minimals


def beat_points(
    topology: FiniteTopology,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Find all beat points (up and down) of a finite topology.

    A point ``x`` is a down beat point if the set of points strictly below x
    has a unique maximal element.
    A point ``x`` is an up beat point if the set of points strictly above x
    has a unique minimal element.
    """
    n = topology.point_count
    preorder = specialization_preorder(topology)
    down_beats: list[int] = []
    up_beats: list[int] = []
    for x in range(n):
        below_x: set[int] = set()
        above_x: set[int] = set()
        for j in range(n):
            if j != x and preorder[x][j]:
                below_x.add(j)
            if j != x and preorder[j][x]:
                above_x.add(j)
        if below_x:
            maximals = _maximal_elements(below_x, preorder)
            if len(maximals) == 1:
                down_beats.append(x)
        if above_x:
            minimals = _minimal_elements(above_x, preorder)
            if len(minimals) == 1:
                up_beats.append(x)
    return (tuple(down_beats), tuple(up_beats))
