"""Known-answer and adversarial tests for finite hypergraph operations."""

from typing import TypedDict

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    CliqueExpansionRequest,
    CliqueExpansionResult,
    DualRequest,
    FiniteHypergraph,
    IncidenceGraphRequest,
    ParametersRequest,
    VertexDegreesRequest,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._operations import (
    compute_clique_expansion,
    compute_dual,
    compute_incidence_graph,
    compute_parameters,
    compute_vertex_degrees,
)
from jacobian.math.graphs.independence import independence_number
from jacobian.math.graphs.values import SimpleUndirectedGraph

# ---- Fixtures ----------------------------------------------------------------

type HyperedgeWire = list[str | list[str]]


class HypergraphWire(TypedDict):
    """Raw JSON-shaped input accepted by ``FiniteHypergraph``."""

    vertices: list[str]
    edges: list[HyperedgeWire]


def _hypergraph(wire: HypergraphWire) -> FiniteHypergraph:
    """Cross the raw-wire boundary into the canonical hypergraph model."""

    return FiniteHypergraph.model_validate(wire)


HYPERGRAPH: HypergraphWire = {
    "vertices": ["a", "b", "c", "d"],
    "edges": [
        ["e1", ["a", "b", "c"]],
        ["e2", ["b", "c", "d"]],
        ["e3", ["a", "d"]],
    ],
}

# Uniform hypergraph: every edge has 2 vertices.
UNIFORM: HypergraphWire = {
    "vertices": ["x", "y", "z"],
    "edges": [
        ["p", ["x", "y"]],
        ["q", ["y", "z"]],
        ["r", ["x", "z"]],
    ],
}

# Hypergraph with no edges.
NO_EDGES: HypergraphWire = {
    "vertices": ["a", "b"],
    "edges": [],
}

# Singleton vertex with one edge.
SINGLETON: HypergraphWire = {
    "vertices": ["v"],
    "edges": [["e", ["v"]]],
}

# Declared vertex order disagrees with lexical order (issue #2300).
REVERSED_ORDER: HypergraphWire = {
    "vertices": ["z", "a"],
    "edges": [["e", ["z", "a"]]],
}


class TestFiniteHypergraph:
    def test_construct(self) -> None:
        hg = _hypergraph(HYPERGRAPH)
        assert hg.vertices == ("a", "b", "c", "d")
        assert hg.edges[0] == ("e1", ("a", "b", "c"))

    def test_member_order_canonicalized(self) -> None:
        hg = _hypergraph({"vertices": ["a", "b"], "edges": [["e", ["b", "a"]]]})
        assert hg.edges == (("e", ("a", "b")),)
        hg2 = _hypergraph({"vertices": ["a", "b"], "edges": [["e", ["a", "b"]]]})
        assert hg == hg2

    def test_duplicate_vertices_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _hypergraph({"vertices": ["a", "a"], "edges": []})

    def test_undeclared_member_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _hypergraph({"vertices": ["a", "b"], "edges": [["e", ["a", "z"]]]})

    def test_duplicate_edge_ids_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _hypergraph(
                {
                    "vertices": ["a", "b"],
                    "edges": [["e", ["a"]], ["e", ["b"]]],
                }
            )

    def test_duplicate_members_in_edge_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _hypergraph({"vertices": ["a", "b"], "edges": [["e", ["a", "a"]]]})

    def test_lone_surrogate_vertex_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _hypergraph({"vertices": ["\ud800"], "edges": []})

    def test_lone_surrogate_edge_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _hypergraph({"vertices": ["a"], "edges": [["\udfff", ["a"]]]})

    def test_astral_plane_label_accepted(self) -> None:
        hg = _hypergraph({"vertices": ["\U0001d5a0"], "edges": [["e", ["\U0001d5a0"]]]})
        assert hg.vertices == ("\U0001d5a0",)
        assert hg.edges[0][1] == ("\U0001d5a0",)


class TestParameters:
    def test_basic(self) -> None:
        r = compute_parameters(ParametersRequest(hypergraph=_hypergraph(HYPERGRAPH)))
        assert r.vertex_count == 4
        assert r.edge_count == 3
        assert r.rank == 3
        assert r.corank == 2
        assert r.uniform_size is None
        assert r.total_incidences == 8

    def test_uniform(self) -> None:
        r = compute_parameters(ParametersRequest(hypergraph=_hypergraph(UNIFORM)))
        assert r.uniform_size == 2
        assert r.rank == 2
        assert r.corank == 2

    def test_no_edges(self) -> None:
        r = compute_parameters(ParametersRequest(hypergraph=_hypergraph(NO_EDGES)))
        assert r.vertex_count == 2
        assert r.edge_count == 0
        assert r.rank == 0
        assert r.corank == 0
        assert r.uniform_size is None
        assert r.total_incidences == 0

    def test_singleton(self) -> None:
        r = compute_parameters(ParametersRequest(hypergraph=_hypergraph(SINGLETON)))
        assert r.vertex_count == 1
        assert r.edge_count == 1
        assert r.rank == 1
        assert r.corank == 1
        assert r.uniform_size == 1


class TestVertexDegrees:
    def test_degrees(self) -> None:
        r = compute_vertex_degrees(
            VertexDegreesRequest(hypergraph=_hypergraph(HYPERGRAPH))
        )
        d = dict(r.degrees)
        assert d["a"] == 2  # e1, e3
        assert d["b"] == 2  # e1, e2
        assert d["c"] == 2  # e1, e2
        assert d["d"] == 2  # e2, e3
        assert dict(r.histogram) == {2: 4}

    def test_uniform(self) -> None:
        r = compute_vertex_degrees(
            VertexDegreesRequest(hypergraph=_hypergraph(UNIFORM))
        )
        assert dict(r.degrees) == {"x": 2, "y": 2, "z": 2}

    def test_no_edges(self) -> None:
        r = compute_vertex_degrees(
            VertexDegreesRequest(hypergraph=_hypergraph(NO_EDGES))
        )
        assert dict(r.degrees) == {"a": 0, "b": 0}
        assert dict(r.histogram) == {0: 2}

    def test_histogram_sorted(self) -> None:
        hg: HypergraphWire = {
            "vertices": ["a", "b", "c", "d"],
            "edges": [["e", ["a", "b"]]],
        }
        r = compute_vertex_degrees(VertexDegreesRequest(hypergraph=_hypergraph(hg)))
        assert dict(r.histogram) == {0: 2, 1: 2}
        degrees = [d for _, d in r.histogram]
        assert degrees == sorted(degrees)

    def test_degrees_in_vertex_order(self) -> None:
        r = compute_vertex_degrees(
            VertexDegreesRequest(hypergraph=_hypergraph(HYPERGRAPH))
        )
        assert [v for v, _ in r.degrees] == ["a", "b", "c", "d"]


class TestDual:
    def test_dual_basic(self) -> None:
        r = compute_dual(DualRequest(hypergraph=_hypergraph(HYPERGRAPH)))
        dual = r.dual
        assert dual.vertices == ("e1", "e2", "e3")
        d_edges = dict(dual.edges)
        assert d_edges["a"] == ("e1", "e3")
        assert d_edges["b"] == ("e1", "e2")
        assert d_edges["c"] == ("e1", "e2")
        assert d_edges["d"] == ("e2", "e3")

    def test_dual_of_dual_recovers_original(self) -> None:
        r = compute_dual(DualRequest(hypergraph=_hypergraph(HYPERGRAPH)))
        r2 = compute_dual(DualRequest(hypergraph=r.dual))
        recovered = r2.dual
        assert recovered.vertices == ("a", "b", "c", "d")
        assert dict(recovered.edges) == {
            "e1": ("a", "b", "c"),
            "e2": ("b", "c", "d"),
            "e3": ("a", "d"),
        }

    def test_dual_no_edges(self) -> None:
        r = compute_dual(DualRequest(hypergraph=_hypergraph(NO_EDGES)))
        assert r.dual.vertices == ()
        assert r.dual.edges == (("a", ()), ("b", ()))


class TestIncidenceGraph:
    def test_incidence(self) -> None:
        r = compute_incidence_graph(
            IncidenceGraphRequest(hypergraph=_hypergraph(HYPERGRAPH))
        )
        vi = dict(r.vertex_incidence)
        assert vi["a"] == ("e1", "e3")
        assert vi["b"] == ("e1", "e2")
        assert vi["c"] == ("e1", "e2")
        assert vi["d"] == ("e2", "e3")
        ei = dict(r.edge_incidence)
        assert ei["e1"] == ("a", "b", "c")
        assert ei["e2"] == ("b", "c", "d")
        assert ei["e3"] == ("a", "d")
        assert set(r.edges) == {
            ("a", "e1"),
            ("a", "e3"),
            ("b", "e1"),
            ("b", "e2"),
            ("c", "e1"),
            ("c", "e2"),
            ("d", "e2"),
            ("d", "e3"),
        }

    def test_no_edges(self) -> None:
        r = compute_incidence_graph(
            IncidenceGraphRequest(hypergraph=_hypergraph(NO_EDGES))
        )
        assert dict(r.vertex_incidence) == {"a": (), "b": ()}
        assert r.edge_incidence == ()
        assert r.edges == ()

    def test_edge_incidence_is_canonical(self) -> None:
        hg = _hypergraph(
            {
                "vertices": ["a", "b", "c"],
                "edges": [["e", ["c", "a", "b"]]],
            }
        )
        r = compute_incidence_graph(IncidenceGraphRequest(hypergraph=hg))
        assert dict(r.edge_incidence)["e"] == ("a", "b", "c")

    def test_vertex_incidence_preserves_edge_order(self) -> None:
        hg: HypergraphWire = {
            "vertices": ["v"],
            "edges": [
                ["z", ["v"]],
                ["a", ["v"]],
                ["m", ["v"]],
            ],
        }
        r = compute_incidence_graph(IncidenceGraphRequest(hypergraph=_hypergraph(hg)))
        assert dict(r.vertex_incidence)["v"] == ("z", "a", "m")


class TestCliqueExpansion:
    def test_canonical_graph(self) -> None:
        r = compute_clique_expansion(
            CliqueExpansionRequest(hypergraph=_hypergraph(HYPERGRAPH))
        )
        assert isinstance(r.graph, SimpleUndirectedGraph)
        assert r.graph.vertices == ("a", "b", "c", "d")
        assert set(r.graph.edges) == {
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
        }

    def test_endpoint_order_follows_graph_convention(self) -> None:
        """The ('z', 'a') reproduction: endpoints are lexical, not declared."""

        r = compute_clique_expansion(
            CliqueExpansionRequest(hypergraph=_hypergraph(REVERSED_ORDER))
        )
        assert r.graph.vertices == ("z", "a")
        assert r.graph.edges == (("a", "z"),)

    def test_defining_property_against_source(self) -> None:
        for hypergraph in (HYPERGRAPH, UNIFORM):
            hg = _hypergraph(hypergraph)
            r = compute_clique_expansion(CliqueExpansionRequest(hypergraph=hg))
            members = [set(members) for _, members in hg.edges]
            adjacent = set(r.graph.edges)
            vertices = hg.vertices
            for i, u in enumerate(vertices):
                for v in vertices[i + 1 :]:
                    shared = any(u in m and v in m for m in members)
                    pair = (min(u, v), max(u, v))
                    assert (pair in adjacent) == shared

    def test_degenerate_edges_admitted(self) -> None:
        hg: HypergraphWire = {
            "vertices": ["a", "b", "c"],
            "edges": [
                ["empty", []],
                ["one", ["a"]],
                ["dup1", ["a", "b"]],
                ["dup2", ["b", "a"]],
                ["pair", ["b", "c"]],
            ],
        }
        r = compute_clique_expansion(CliqueExpansionRequest(hypergraph=_hypergraph(hg)))
        assert r.graph.vertices == ("a", "b", "c")
        assert r.graph.edges == (("a", "b"), ("b", "c"))

    def test_no_edges(self) -> None:
        r = compute_clique_expansion(
            CliqueExpansionRequest(hypergraph=_hypergraph(NO_EDGES))
        )
        assert r.graph.vertices == ("a", "b")
        assert r.graph.edges == ()

    def test_singleton(self) -> None:
        r = compute_clique_expansion(
            CliqueExpansionRequest(hypergraph=_hypergraph(SINGLETON))
        )
        assert r.graph.vertices == ("v",)
        assert r.graph.edges == ()

    def test_non_nfc_vertex_label_rejected(self) -> None:
        decomposed = "e\u0301"
        request = CliqueExpansionRequest.model_validate(
            {
                "hypergraph": {
                    "vertices": [decomposed, "a"],
                    "edges": [["e", [decomposed, "a"]]],
                }
            }
        )
        with pytest.raises(OperationDomainValidationError):
            compute_clique_expansion(request)

    def test_nfc_vertex_label_accepted(self) -> None:
        composed = "\u00e9"
        r = compute_clique_expansion(
            CliqueExpansionRequest.model_validate(
                {
                    "hypergraph": {
                        "vertices": [composed, "a"],
                        "edges": [["e", [composed, "a"]]],
                    }
                }
            )
        )
        assert r.graph.vertices == ("\u00e9", "a")
        assert r.graph.edges == (("a", "\u00e9"),)

    def test_dual_expansion_independence_composition(self) -> None:
        """Dual -> clique expansion -> independence number, with no rewriting.

        The expansion of the dual is the intersection graph on the original
        indexed hyperedges; a maximum independent set is exactly a maximum
        pairwise-disjoint family of source hyperedges.
        """

        hg = _hypergraph(
            {
                "vertices": ["a", "b", "c", "d", "e", "f"],
                "edges": [
                    ["e1", ["a", "b"]],
                    ["e2", ["c", "d"]],
                    ["e3", ["e", "f"]],
                    ["e4", ["a", "c"]],
                ],
            }
        )
        dual = compute_dual(DualRequest(hypergraph=hg)).dual
        expansion = compute_clique_expansion(CliqueExpansionRequest(hypergraph=dual))

        result = independence_number(expansion.graph)
        assert result.status == "EXACT"
        assert result.optimum_value == 3

        members = dict(hg.edges)
        witnesses = result.witness_vertices
        assert len(witnesses) == 3
        for i, u in enumerate(witnesses):
            for v in witnesses[i + 1 :]:
                assert not set(members[u]) & set(members[v])

    def test_serialization_round_trip(self) -> None:
        r = compute_clique_expansion(
            CliqueExpansionRequest(hypergraph=_hypergraph(HYPERGRAPH))
        )
        restored = CliqueExpansionResult.model_validate_json(r.model_dump_json())
        assert restored == r


class TestBindingSafety:
    def test_clique_expansion_binding_rejects_declared_order_endpoints(self) -> None:
        with pytest.raises(ValidationError):
            SimpleUndirectedGraph(vertices=("z", "a"), edges=(("z", "a"),))
