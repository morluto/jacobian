"""Defining-invariant and boundary tests for the edge-intersection graph."""

import pytest
from pydantic import ValidationError

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
    def test_retains_source_edge_ids(self) -> None:
        """Graph vertices are exactly the source edge IDs in declared order."""

        r = _graph(FIXTURE)

        assert r.graph.vertices == tuple(edge_id for edge_id, _ in r.hypergraph.edges)

    def test_serialization_round_trip(self) -> None:
        r = _graph(FIXTURE)

        restored = EdgeIntersectionGraphResult.model_validate_json(r.model_dump_json())
        assert restored == r

    def test_deserialization_binds_graph_axis_to_source_edges(self) -> None:
        source = FiniteHypergraph.model_validate(FIXTURE)
        graph = SimpleUndirectedGraph(
            vertices=("E1", "E0", "E2"),
            edges=(),
        )

        with pytest.raises(ValidationError) as caught:
            EdgeIntersectionGraphResult(hypergraph=source, graph=graph)

        assert caught.value.errors()[0]["type"] == "hypergraph.source_axis"


class TestEdgeIntersectionGraphBoundary:
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
        ("wire", "expected_vertices", "expected_edges"),
        [
            (FIXTURE, ("E0", "E1", "E2"), (("E0", "E1"),)),
            ({"vertices": ["a", "b"], "edges": []}, (), ()),
            (
                {"vertices": ["a", "b"], "edges": [["e1", ["a"]], ["e2", ["b"]]]},
                ("e1", "e2"),
                (),
            ),
            (
                {"vertices": ["a", "b"], "edges": [["only", ["a", "b"]]]},
                ("only",),
                (),
            ),
            (
                {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["e1", ["a", "b"]], ["e2", ["c", "d"]]],
                },
                ("e1", "e2"),
                (),
            ),
            (
                {
                    "vertices": ["a", "b", "c"],
                    "edges": [
                        ["e1", ["a", "b"]],
                        ["e2", ["a", "c"]],
                        ["e3", ["a"]],
                    ],
                },
                ("e1", "e2", "e3"),
                (("e1", "e2"), ("e1", "e3"), ("e2", "e3")),
            ),
            (
                {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [
                        ["e1", ["a", "b", "c"]],
                        ["e2", ["b", "c", "d"]],
                        ["e3", ["a", "d"]],
                    ],
                },
                ("e1", "e2", "e3"),
                (("e1", "e2"), ("e1", "e3"), ("e2", "e3")),
            ),
            (
                {
                    "vertices": ["a", "b", "c"],
                    "edges": [
                        ["z", ["a", "b"]],
                        ["a", ["b", "c"]],
                        ["m", ["a", "c"]],
                    ],
                },
                ("z", "a", "m"),
                (("a", "m"), ("a", "z"), ("m", "z")),
            ),
            (
                {
                    "vertices": ["a", "b", "c", "d", "e", "f"],
                    "edges": [
                        ["e1", ["a", "b"]],
                        ["e2", ["c", "d"]],
                        ["e3", ["e", "f"]],
                        ["e4", ["a", "c"]],
                    ],
                },
                ("e1", "e2", "e3", "e4"),
                (("e1", "e4"), ("e2", "e4")),
            ),
        ],
    )
    def test_edge_intersection_defining_property(
        self,
        wire: object,
        expected_vertices: tuple[str, ...],
        expected_edges: tuple[tuple[str, str], ...],
    ) -> None:
        r = _graph(wire)
        assert r.graph.vertices == expected_vertices
        assert set(r.graph.edges) == set(expected_edges)

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

    def test_sparse_graph_is_admitted_by_its_actual_output_size(self) -> None:
        """A large sparse graph is not charged for absent complete-graph edges."""
        edges = [[f"e{i:03}", [f"v{i}"]] for i in range(256)]
        edges[1][1] = ["v0"]
        result = _graph({"vertices": [f"v{i}" for i in range(256)], "edges": edges})
        assert result.graph.edges == (("e000", "e001"),)
