"""Pure admission predicates shared by graphical-model entry points."""

from __future__ import annotations

from collections import deque

from jacobian.math.graphical_models.values import MAX_MODEL_VARS


def validate_d_separation_input(
    variable_count: int,
    edges: tuple[tuple[int, int], ...],
    set_a: tuple[int, ...],
    set_b: tuple[int, ...],
    set_c: tuple[int, ...],
) -> None:
    """Validate one bounded DAG and its pairwise-disjoint node sets."""

    if not 1 <= variable_count <= MAX_MODEL_VARS:
        raise ValueError("variable_count must be between 1 and 16")
    if len(set(edges)) != len(edges):
        raise ValueError("directed edges must be distinct")
    parents: dict[int, set[int]] = {node: set() for node in range(variable_count)}
    children: dict[int, set[int]] = {node: set() for node in range(variable_count)}
    for parent, child in edges:
        if not 0 <= parent < variable_count or not 0 <= child < variable_count:
            raise ValueError("edge endpoint is outside the graph")
        if parent == child:
            raise ValueError("directed graph cannot contain a self-loop")
        parents[child].add(parent)
        children[parent].add(child)
    indegree = {node: len(parents[node]) for node in parents}
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != variable_count:
        raise ValueError("d-separation requires a directed acyclic graph")
    node_sets = (set_a, set_b, set_c)
    if not set_a or not set_b:
        raise ValueError("sets A and B must be nonempty")
    if any(len(values) != len(set(values)) for values in node_sets):
        raise ValueError("d-separation node sets cannot contain duplicates")
    if any(not 0 <= node < variable_count for values in node_sets for node in values):
        raise ValueError("d-separation node is outside the graph")
    if set(set_a) & set(set_b) or set(set_a) & set(set_c) or set(set_b) & set(set_c):
        raise ValueError("d-separation node sets must be pairwise disjoint")
