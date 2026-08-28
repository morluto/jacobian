"""Defining-invariant and boundary tests for minimum transversals."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
    MinimumTransversalRequest,
    MinimumTransversalResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._operations import (
    compute_minimum_transversal,
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
    return compute_minimum_transversal(
        MinimumTransversalRequest(hypergraph=FiniteHypergraph.model_validate(source))
    )


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

    def test_request_reserves_retained_source_and_witness_output_bytes(self) -> None:
        vertices = [f"v{i:03d}" + "😀" * 60 for i in range(256)]
        edge_member_sets = [
            (vertices[0], vertices[2 * i + 1], vertices[2 * i + 2]) for i in range(127)
        ]
        edge_member_sets.append((vertices[0], vertices[1], vertices[255]))
        edges = [
            (f"{i:05d}" + "🚀" * 28, edge_member_sets[i % 128]) for i in range(12_000)
        ]
        hypergraph = FiniteHypergraph(vertices=tuple(vertices), edges=tuple(edges))

        source_bytes = len(canonicalize_json(hypergraph.model_dump(mode="json")))
        assert source_bytes < CanonicalLimits().max_output_bytes
        with pytest.raises(ValueError, match="canonical output limit"):
            _transversal(hypergraph)

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
        vertices = [f"v{i}" for i in range(22)]
        hg = {
            "vertices": vertices,
            "edges": [[f"e{i}", [vertex]] for i, vertex in enumerate(vertices)],
        }
        with pytest.raises(ValueError):
            _transversal(hg)

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
