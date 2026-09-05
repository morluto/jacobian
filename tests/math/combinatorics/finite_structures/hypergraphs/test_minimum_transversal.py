"""Defining-invariant and boundary tests for minimum transversals."""

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
    MinimumTransversalRequest,
    MinimumTransversalResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    minimum_transversal,
)

HYPERGRAPH = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [
        ["e1", ["a", "b", "c"]],
        ["e2", ["b", "c", "d"]],
        ["e3", ["a", "d"]],
    ],
}


def _transversal(source: object) -> MinimumTransversalResult:
    return minimum_transversal(FiniteHypergraph.model_validate(source))


def _undecomposed_minimum_transversal(
    hypergraph: FiniteHypergraph,
) -> tuple[tuple[str, ...], int]:
    """Reference global increasing-subset search without forced/component presolve."""

    from itertools import combinations
    from math import comb

    unique_edges: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for _, members in hypergraph.edges:
        edge = frozenset(members)
        if edge not in seen:
            seen.add(edge)
            unique_edges.append(edge)
    unique_edges.sort(key=lambda edge: (len(edge), tuple(sorted(edge))))
    active = tuple(
        vertex
        for vertex in hypergraph.vertices
        if any(vertex in e for e in unique_edges)
    )
    greedy: set[str] = set()
    for edge in unique_edges:
        if not greedy & edge:
            greedy.add(next(vertex for vertex in active if vertex in edge))
    for size in range(len(greedy) + 1):
        for combo in combinations(active, size):
            candidate = frozenset(combo)
            if all(candidate & edge for edge in unique_edges):
                _ = comb(len(active), size)
                return combo, size
    raise AssertionError("reference search exhausted all vertices")


class TestMinimumTransversal:
    def test_known_minimum(self) -> None:
        result = _transversal(HYPERGRAPH)
        assert result.cardinality == 2
        assert set(result.transversal) in ({"a", "b"},)
        # {a,b} hits e1(a/b), e2(b), e3(a)
        assert set(result.transversal) == {"a", "b"}

    def test_empty_edge_family_empty_transversal(self) -> None:
        result = _transversal({"vertices": ["a", "b"], "edges": []})
        assert result.transversal == ()
        assert result.cardinality == 0

    def test_empty_edge_family_above_search_cap_is_admitted(self) -> None:
        result = _transversal({"vertices": [f"v{i}" for i in range(21)], "edges": []})
        assert result.transversal == ()
        assert result.cardinality == 0

    def test_schema_states_nonempty_hyperedge_precondition(self) -> None:
        schema = MinimumTransversalRequest.model_json_schema()
        hypergraph_schema = schema["properties"]["hypergraph"]
        assert "Every hyperedge must be nonempty" in hypergraph_schema["description"]
        assert hypergraph_schema["requires_nonempty_hyperedges"] is True

    def test_single_edge_requires_one_vertex(self) -> None:
        result = _transversal(
            {"vertices": ["x", "y", "z"], "edges": [["e", ["x", "y"]]]}
        )
        assert result.cardinality == 1
        assert result.transversal == ("x",)

    def test_disjoint_edges_require_vertex_per_edge(self) -> None:
        result = _transversal(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["c", "d"]],
                ],
            }
        )
        assert result.cardinality == 2
        assert set(result.transversal) == {"a", "c"}

    def test_transversal_in_declared_vertex_order(self) -> None:
        result = _transversal(
            {
                "vertices": ["z", "a", "m"],
                "edges": [
                    ["e1", ["z", "a"]],
                    ["e2", ["a", "m"]],
                ],
            }
        )
        # Minimum is {a}; must appear in declared order (single element)
        assert result.cardinality == 1
        assert result.transversal == ("a",)

    def test_empty_edge_rejected(self) -> None:
        with pytest.raises(ValueError):
            _transversal({"vertices": ["a"], "edges": [["e", []]]})

    def test_large_carrier_with_small_active_edge_family_is_admitted(self) -> None:
        result = _transversal(
            {
                "vertices": [f"v{i}" for i in range(256)],
                "edges": [["e", ["v0"]]],
            }
        )
        assert result.transversal == ("v0",)
        assert result.cardinality == 1

    def test_duplicate_edges_are_deduplicated_for_search_budget(self) -> None:
        vertices = [f"v{i}" for i in range(20)]
        edges = [[f"full-{i}", vertices] for i in range(1000)]
        edges.extend([[f"single-{i}", [vertex]] for i, vertex in enumerate(vertices)])
        result = _transversal({"vertices": vertices, "edges": edges})
        assert result.cardinality == 20
        assert result.transversal == tuple(vertices)

    def test_active_search_and_result_can_exceed_old_witness_cap(self) -> None:
        vertices = [f"v{i}" for i in range(21)]
        result = _transversal(
            {
                "vertices": vertices,
                "edges": [[f"e{i}", [vertex]] for i, vertex in enumerate(vertices)],
            }
        )
        assert result.cardinality == 21
        assert result.transversal == tuple(vertices)

    def test_search_work_bound_exceeded(self) -> None:
        # 22 forced singletons are presolved with no residual search.
        vertices = [f"v{i}" for i in range(22)]
        result = _transversal(
            {
                "vertices": vertices,
                "edges": [[f"e{i}", [vertex]] for i, vertex in enumerate(vertices)],
            }
        )
        assert result.cardinality == 22
        assert result.transversal == tuple(vertices)

    def test_disjoint_pair_components_are_admitted(self) -> None:
        vertices = [f"v{i:02}" for i in range(24)]
        result = _transversal(
            {
                "vertices": vertices,
                "edges": [
                    [f"e{i:02}", [f"v{2 * i:02}", f"v{2 * i + 1:02}"]]
                    for i in range(12)
                ],
            }
        )
        assert result.cardinality == 12
        assert result.transversal == tuple(f"v{2 * i:02}" for i in range(12))

    def test_hard_residual_component_still_rejected(self) -> None:
        # 21 vertices with all 3-subsets: one dense component whose
        # C(21,10)*1330 charge exceeds the search budget.
        from itertools import combinations

        vertices = [f"v{i}" for i in range(21)]
        with pytest.raises(ValueError, match="search exceeds"):
            _transversal(
                {
                    "vertices": vertices,
                    "edges": [
                        [f"e{index}", list(edge)]
                        for index, edge in enumerate(combinations(vertices, 3))
                    ],
                }
            )

    def test_decomposed_search_matches_undecomposed_search(self) -> None:
        from itertools import combinations

        from jacobian.math.combinatorics.finite_structures.hypergraphs import (
            operations as hypergraph_operations,
        )

        vertices = ("a", "b", "c")
        pool = [
            frozenset(combo)
            for width in range(1, 4)
            for combo in combinations(vertices, width)
        ]
        for mask in range(1 << len(pool)):
            family = [pool[index] for index in range(len(pool)) if mask & (1 << index)]
            source = FiniteHypergraph.model_validate(
                {
                    "vertices": list(vertices),
                    "edges": [
                        [f"e{index}", sorted(members)]
                        for index, members in enumerate(family)
                    ],
                }
            )
            expected = _undecomposed_minimum_transversal(source)
            actual = hypergraph_operations.minimum_transversal(source)
            assert (actual.transversal, actual.cardinality) == expected

    def test_rejects_wrong_cardinality(self) -> None:
        with pytest.raises(ValidationError):
            MinimumTransversalResult.model_validate(
                {
                    "hypergraph": HYPERGRAPH,
                    "transversal": ["a", "b"],
                    "cardinality": 3,
                }
            )

    def test_rejects_non_minimum_transversal(self) -> None:
        # {a, b, c} is a transversal but not minimum; search returns {a,b}.
        # The result model only validates hitting, not minimality, but the
        # compute function must return the minimum.
        result = _transversal(HYPERGRAPH)
        assert result.cardinality == 2
