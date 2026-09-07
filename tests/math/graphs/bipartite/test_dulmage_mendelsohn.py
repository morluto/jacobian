"""DM evidence from enumeration of all maximum matchings, not a second solver."""

import json
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.bipartite import FixedBipartiteGraph, dulmage_mendelsohn
from jacobian.math.graphs.bipartite._models import DulmageMendelsohnDecomposition
from jacobian.math.graphs.bipartite._tools import TOOLS


def source(p: int, q: int, edges: Sequence[tuple[int, int]]) -> FixedBipartiteGraph:
    return FixedBipartiteGraph.model_validate(
        {
            "graph": {
                "vertex_count": p + q,
                "edges": tuple(sorted((i, p + j) for i, j in edges)),
            },
            "left_vertices": tuple(range(p)),
            "right_vertices": tuple(range(p, p + q)),
        }
    )


def all_maximum_matchings(
    graph: FixedBipartiteGraph,
) -> list[frozenset[tuple[int, int]]]:
    neighbors: dict[int, list[int]] = {v: [] for v in graph.left_vertices}
    for a, b in graph.graph.edges:
        u, v = (a, b) if a in neighbors else (b, a)
        neighbors[u].append(v)
    matchings: list[frozenset[tuple[int, int]]] = []

    def enumerate_at(k: int, chosen: list[tuple[int, int]], used: set[int]) -> None:
        if k == len(graph.left_vertices):
            matchings.append(frozenset(chosen))
            return
        u = graph.left_vertices[k]
        enumerate_at(k + 1, chosen, used)
        for v in neighbors[u]:
            if v not in used:
                enumerate_at(k + 1, [*chosen, (u, v)], used | {v})

    enumerate_at(0, [], set())
    rank = max(map(len, matchings))
    return [matching for matching in matchings if len(matching) == rank]


def check_oracle(graph: FixedBipartiteGraph) -> DulmageMendelsohnDecomposition:
    matchings = all_maximum_matchings(graph)
    result = dulmage_mendelsohn(graph)
    assert result.structural_rank == len(matchings[0])
    # A left vertex is in the left-excess region iff some maximum matching
    # leaves it exposed. Its neighbors are exactly that region's right side.
    exposed_left = set().union(
        *(set(graph.left_vertices) - {u for u, _ in matching} for matching in matchings)
    )
    exposed_right = set().union(
        *(
            set(graph.right_vertices) - {v for _, v in matching}
            for matching in matchings
        )
    )
    left = set(graph.left_vertices)
    oriented = [(a, b) if a in left else (b, a) for a, b in graph.graph.edges]
    right_of_left = {v for u, v in oriented if u in exposed_left}
    left_of_right = {u for u, v in oriented if v in exposed_right}
    assert set(result.left_excess.left_vertices) == exposed_left
    assert set(result.left_excess.right_vertices) == right_of_left
    assert set(result.right_excess.left_vertices) == left_of_right
    assert set(result.right_excess.right_vertices) == exposed_right
    balanced = (
        set(range(graph.graph.vertex_count))
        - exposed_left
        - exposed_right
        - right_of_left
        - left_of_right
    )
    allowed = set().union(*matchings)
    # On the balanced part, connected components of edges appearing in
    # ANY maximum matching are the elementary DM blocks. This oracle uses
    # undirected union-find, not the production alternating SCC construction.
    roots = {v: v for v in balanced}

    def root(v: int) -> int:
        while roots[v] != v:
            v = roots[v]
        return v

    for u, v in allowed:
        if u in balanced and v in balanced:
            roots[root(u)] = root(v)
    components = {
        frozenset(v for v in balanced if root(v) == root(u)) for u in balanced
    }
    actual = [
        set(block.left_vertices) | set(block.right_vertices)
        for block in result.balanced_blocks
    ]
    assert {frozenset(block) for block in actual} == components
    index = {v: i for i, block in enumerate(actual) for v in block}
    expected_arcs = {
        (index[u], index[v])
        for u, v in oriented
        if u in balanced and v in balanced and index[u] != index[v]
    }
    assert set(result.condensation_edges) == expected_arcs
    assert (
        DulmageMendelsohnDecomposition.model_validate_json(result.model_dump_json())
        == result
    )
    return result


def test_all_patterns_through_three_by_three() -> None:
    for p in range(4):
        for q in range(4):
            possible = [(i, j) for i in range(p) for j in range(q)]
            for mask in range(1 << len(possible)):
                check_oracle(
                    source(
                        p, q, [edge for k, edge in enumerate(possible) if mask >> k & 1]
                    )
                )


def test_triangular_order_and_complete_elementary_block() -> None:
    triangular = check_oracle(source(2, 2, [(0, 0), (0, 1), (1, 1)]))
    assert [block.left_vertices for block in triangular.balanced_blocks] == [(0,), (1,)]
    assert triangular.condensation_edges == ((0, 1),)
    complete = check_oracle(source(2, 2, [(i, j) for i in range(2) for j in range(2)]))
    assert len(complete.balanced_blocks) == 1


def test_union_of_opposite_deficiencies_balanced_and_isolates() -> None:
    result = check_oracle(source(5, 5, [(0, 0), (1, 0), (2, 1), (2, 2), (3, 3)]))
    assert result.left_excess.left_vertices == (0, 1, 4)
    assert result.right_excess.right_vertices == (6, 7, 9)
    assert result.balanced_blocks[0].left_vertices == (3,)


def test_relabelled_and_reordered_sides_preserve_source_order() -> None:
    original = source(3, 3, [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 2)])
    perm = {0: 5, 1: 2, 2: 4, 3: 1, 4: 3, 5: 0}
    relabelled = FixedBipartiteGraph.model_validate(
        {
            "graph": {
                "vertex_count": 6,
                "edges": tuple(
                    sorted(
                        tuple(sorted((perm[a], perm[b])))
                        for a, b in original.graph.edges
                    )
                ),
            },
            "left_vertices": tuple(perm[v] for v in original.left_vertices),
            "right_vertices": tuple(perm[v] for v in original.right_vertices),
        }
    )
    first, second = check_oracle(original), check_oracle(relabelled)
    assert second.condensation_edges == first.condensation_edges
    for a, b in zip(first.balanced_blocks, second.balanced_blocks, strict=True):
        assert b.left_vertices == tuple(perm[v] for v in a.left_vertices)
        assert b.right_vertices == tuple(perm[v] for v in a.right_vertices)


@pytest.mark.parametrize(
    "left,right,edges",
    [
        ((0,), (0,), ()),
        ((0,), (), ()),
        ((0, 1), (), ((0, 1),)),
        ((0, 0), (1,), ()),
    ],
)
def test_invalid_sides_are_structural_errors(
    left: tuple[int, ...], right: tuple[int, ...], edges: tuple[tuple[int, int], ...]
) -> None:
    with pytest.raises(ValidationError):
        FixedBipartiteGraph.model_validate(
            {
                "graph": {"vertex_count": 2, "edges": edges},
                "left_vertices": left,
                "right_vertices": right,
            }
        )


def test_native_wire_parity() -> None:
    tool = TOOLS[0]
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    assert tool.run(request) == dulmage_mendelsohn(request.graph)


def test_sparse_1024_vertex_triangular_pattern() -> None:
    n = 512
    result = dulmage_mendelsohn(
        source(n, n, [(i, i) for i in range(n)] + [(i, i + 1) for i in range(n - 1)])
    )
    assert result.structural_rank == n
    assert len(result.balanced_blocks) == n
    assert result.condensation_edges == tuple((i, i + 1) for i in range(n - 1))


def test_sparse_carrier_boundary_rejection() -> None:
    with pytest.raises(ValidationError):
        source(513, 512, [(i, i) for i in range(512)])


def test_sparse_and_dense_accepted_boundary() -> None:
    # The full canonical edge envelope remains executable.
    n = 256
    result = dulmage_mendelsohn(
        source(n, n, [(i, j) for i in range(n) for j in range(n)])
    )
    assert result.structural_rank == n
    assert len(result.balanced_blocks) == 1
    assert not result.condensation_edges
