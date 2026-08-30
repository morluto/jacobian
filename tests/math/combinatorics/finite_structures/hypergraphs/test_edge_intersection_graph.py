"""Defining-invariant and boundary tests for the edge-intersection graph."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    EdgeIntersectionGraphRequest,
    EdgeIntersectionGraphResult,
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.operations import (
    edge_intersection_graph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(source: object) -> EdgeIntersectionGraphResult:
    return edge_intersection_graph(FiniteHypergraph.model_validate(source))


# ---- Issue fixture: E0={a,b}, E1={b,c}, E2={d} → only edge E0-E1 -------------

FIXTURE = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [
        ["E0", ["a", "b"]],
        ["E1", ["b", "c"]],
        ["E2", ["d"]],
    ],
}


class TestEdgeIntersectionGraph:
    def test_issue_fixture_vertices_and_single_edge(self) -> None:
        """The issue fixture: vertices E0,E1,E2 and exactly the edge E0-E1."""

        r = _graph(FIXTURE)

        assert isinstance(r.graph, SimpleUndirectedGraph)
        assert r.graph.vertices == ("E0", "E1", "E2")
        assert r.graph.edges == (("E0", "E1"),)

    def test_defining_property_every_edge_is_nonempty_intersection(
        self,
    ) -> None:
        """Replay every graph edge as a nonempty source-edge intersection."""

        r = _graph(FIXTURE)

        member_map = dict(r.hypergraph.edges)
        for u, v in r.graph.edges:
            assert set(member_map[u]) & set(member_map[v]), (
                f"graph edge ({u}, {v}) must come from a nonempty intersection"
            )

    def test_defining_property_every_omitted_pair_is_disjoint(self) -> None:
        """Every omitted pair must be disjoint source hyperedges."""

        r = _graph(FIXTURE)

        member_map = dict(r.hypergraph.edges)
        vertices = r.graph.vertices
        adjacent = set(r.graph.edges)
        for i, u in enumerate(vertices):
            for v in vertices[i + 1 :]:
                pair = (min(u, v), max(u, v))
                shared = bool(set(member_map[u]) & set(member_map[v]))
                if pair in adjacent:
                    assert shared, f"adjacent pair {pair} must intersect"
                else:
                    assert not shared, f"non-adjacent pair {pair} must be disjoint"

    def test_retains_source_edge_ids(self) -> None:
        """Graph vertices are exactly the source edge IDs in declared order."""

        r = _graph(FIXTURE)

        assert r.graph.vertices == tuple(edge_id for edge_id, _ in r.hypergraph.edges)

    def test_serialization_round_trip(self) -> None:
        r = _graph(FIXTURE)

        restored = EdgeIntersectionGraphResult.model_validate_json(r.model_dump_json())
        assert restored == r


class TestEdgeIntersectionGraphBoundary:
    def test_no_edges_empty_graph(self) -> None:
        r = _graph({"vertices": ["a", "b"], "edges": []})

        assert r.graph.vertices == ()
        assert r.graph.edges == ()

    def test_single_edge_no_adjacency(self) -> None:
        r = _graph(
            {
                "vertices": ["a", "b"],
                "edges": [["only", ["a", "b"]]],
            }
        )

        assert r.graph.vertices == ("only",)
        assert r.graph.edges == ()

    def test_all_disjoint_edges_no_adjacency(self) -> None:
        r = _graph(
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["c", "d"]],
                ],
            }
        )

        assert r.graph.vertices == ("e1", "e2")
        assert r.graph.edges == ()

    def test_duplicate_member_sets_remain_distinct_vertices(self) -> None:
        """Parallel-looking equal sets remain distinct positions."""

        r = _graph(
            {
                "vertices": ["a", "b"],
                "edges": [
                    ["copy-1", ["a", "b"]],
                    ["copy-2", ["a", "b"]],
                ],
            }
        )

        assert r.graph.vertices == ("copy-1", "copy-2")
        assert r.graph.edges == (("copy-1", "copy-2"),)

    def test_empty_edge_ids_are_vertices(self) -> None:
        """Empty hyperedges still produce graph vertices."""

        r = _graph(
            {
                "vertices": ["a"],
                "edges": [
                    ["empty", []],
                    ["full", ["a"]],
                ],
            }
        )

        assert r.graph.vertices == ("empty", "full")
        assert r.graph.edges == ()

    def test_complete_intersection_graph(self) -> None:
        """All edges sharing a common vertex produce a complete graph."""

        r = _graph(
            {
                "vertices": ["a", "b", "c"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["a", "c"]],
                    ["e3", ["a"]],
                ],
            }
        )

        assert r.graph.vertices == ("e1", "e2", "e3")
        assert set(r.graph.edges) == {
            ("e1", "e2"),
            ("e1", "e3"),
            ("e2", "e3"),
        }

    def test_lexical_edge_order_follows_graph_convention(self) -> None:
        r = _graph(
            {
                "vertices": ["a", "b"],
                "edges": [
                    ["z", ["a"]],
                    ["a", ["a"]],
                ],
            }
        )

        assert r.graph.vertices == ("z", "a")
        assert r.graph.edges == (("a", "z"),)


class TestEdgeIntersectionGraphNFC:
    def test_non_nfc_edge_id_rejected(self) -> None:
        decomposed = "e\u0301"
        request = EdgeIntersectionGraphRequest.model_validate(
            {
                "hypergraph": {
                    "vertices": ["a"],
                    "edges": [[decomposed, ["a"]]],
                }
            }
        )
        with pytest.raises(OperationDomainValidationError):
            edge_intersection_graph(request.hypergraph)

    def test_nfc_edge_id_accepted(self) -> None:
        composed = "\u00e9"
        request = EdgeIntersectionGraphRequest.model_validate(
            {
                "hypergraph": {
                    "vertices": ["a"],
                    "edges": [[composed, ["a"]]],
                }
            }
        )
        r = edge_intersection_graph(request.hypergraph)
        assert r.graph.vertices == (composed,)
        assert r.graph.edges == ()


class TestEdgeIntersectionGraphDefiningProperty:
    """Property-based: replay every edge as intersection and every omission as disjoint."""

    @pytest.mark.parametrize(
        "wire",
        [
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["e1", ["a", "b", "c"]],
                    ["e2", ["b", "c", "d"]],
                    ["e3", ["a", "d"]],
                ],
            },
            {
                "vertices": ["a", "b", "c"],
                "edges": [
                    ["z", ["a", "b"]],
                    ["a", ["b", "c"]],
                    ["m", ["a", "c"]],
                ],
            },
            {
                "vertices": ["a", "b", "c", "d", "e", "f"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["c", "d"]],
                    ["e3", ["e", "f"]],
                    ["e4", ["a", "c"]],
                ],
            },
        ],
    )
    def test_edge_intersection_defining_property(self, wire: object) -> None:
        r = _graph(wire)

        member_map = dict(r.hypergraph.edges)
        vertices = r.graph.vertices
        adjacent = set(r.graph.edges)
        for i, u in enumerate(vertices):
            for v in vertices[i + 1 :]:
                pair = (min(u, v), max(u, v))
                shared = bool(set(member_map[u]) & set(member_map[v]))
                assert (pair in adjacent) == shared


class TestEdgeIntersectionGraphCarrierBound:
    """The edge-intersection graph must fit the SimpleUndirectedGraph carrier."""

    def test_too_many_edges_rejected(self) -> None:
        """More than 256 hyperedges produce too many graph vertices."""

        too_many = [
            {"vertices": ["v0"], "edges": [[f"e{i}", ["v0"]] for i in range(257)]}
        ]
        with pytest.raises(OperationDomainValidationError) as exc:
            _graph(too_many[0])
        assert "carrier_vertex_bound" in exc.value.errors()[0]["type"]

    def test_edge_id_too_long_rejected(self) -> None:
        """Edge IDs exceeding 64 characters cannot fit the graph carrier."""

        long_id = "x" * 65
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _graph(
                {
                    "vertices": ["a", "b"],
                    "edges": [
                        [long_id, ["a"]],
                        ["e2", ["b"]],
                    ],
                }
            )
