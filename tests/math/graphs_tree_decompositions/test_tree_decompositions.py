"""Tests for tree-decomposition operations."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.tree_decompositions import TreeDecomposition
from jacobian.math.graphs.tree_decompositions._models import (
    AdhesionsRequest,
    BagIntersectionGraphRequest,
    RerootRequest,
    RestrictRequest,
    VertexOccurrencesRequest,
    WidthRequest,
)
from jacobian.math.graphs.tree_decompositions._operations import (
    compute_adhesions,
    compute_bag_intersection_graph,
    compute_reroot,
    compute_restrict,
    compute_vertex_occurrences,
    compute_width,
)
from jacobian.math.graphs.tree_decompositions._tools import TOOLS
from jacobian.math.graphs.values import SimpleUndirectedGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_graph() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        graph_schema_version="1",
        vertices=("a", "b", "c"),
        edges=(("a", "b"), ("b", "c")),
    )


def _path_decomposition() -> TreeDecomposition:
    """A valid tree decomposition of the path a-b-c with two bags {a,b} and {b,c}."""
    return TreeDecomposition(
        graph=_path_graph(),
        tree_nodes=("t0", "t1"),
        tree_edges=(("t0", "t1"),),
        bags=(("a", "b"), ("b", "c")),
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "graph.tree_decomposition.width.compute",
        "graph.tree_decomposition.vertex_occurrences.compute",
        "graph.tree_decomposition.adhesions.compute",
        "graph.tree_decomposition.reroot.compute",
        "graph.tree_decomposition.restrict.compute",
        "graph.tree_decomposition.bag_intersection_graph.compute",
    }


# ---------------------------------------------------------------------------
# Width
# ---------------------------------------------------------------------------


class TestWidth:
    def test_path_width_is_one(self) -> None:
        result = compute_width(WidthRequest(decomposition=_path_decomposition()))
        assert result.width == 1
        assert result.max_bag_cardinality == 2
        assert result.bag_sizes == (2, 2)
        assert set(result.maximum_bag_nodes) == {"t0", "t1"}

    def test_single_node_decomposition(self) -> None:
        graph = SimpleUndirectedGraph(
            graph_schema_version="1",
            vertices=("a", "b"),
            edges=(("a", "b"),),
        )
        td = TreeDecomposition(
            graph=graph,
            tree_nodes=("t0",),
            tree_edges=(),
            bags=(("a", "b"),),
        )
        assert compute_width(WidthRequest(decomposition=td)).width == 1


# ---------------------------------------------------------------------------
# Vertex occurrences
# ---------------------------------------------------------------------------


class TestVertexOccurrences:
    def test_path_occurrences(self) -> None:
        td = _path_decomposition()
        result = compute_vertex_occurrences(VertexOccurrencesRequest(decomposition=td))
        per_vertex = result.per_vertex
        # Vertex b appears in both bags.
        assert set(cast(Any, per_vertex["b"]["nodes"])) == {"t0", "t1"}
        assert cast(int, per_vertex["b"]["count"]) == 2
        assert cast(Any, per_vertex["b"]["edges"]) == (("t0", "t1"),)
        # Vertex a appears in one bag.
        assert cast(Any, per_vertex["a"]["nodes"]) == ("t0",)
        assert cast(int, per_vertex["a"]["count"]) == 1
        # Vertex c appears in one bag.
        assert per_vertex["c"]["nodes"] == ("t1",)
        assert cast(int, per_vertex["c"]["count"]) == 1


# ---------------------------------------------------------------------------
# Adhesions
# ---------------------------------------------------------------------------


class TestAdhesions:
    def test_path_adhesions(self) -> None:
        td = _path_decomposition()
        result = compute_adhesions(AdhesionsRequest(decomposition=td))
        assert result.max_adhesion == 1
        assert len(result.edges) == 1
        edge = result.edges[0]
        assert edge["edge"] == ("t0", "t1")
        assert edge["adhesion"] == ("b",)
        assert edge["size"] == 1
        assert result.size_profile == (1,)


# ---------------------------------------------------------------------------
# Reroot
# ---------------------------------------------------------------------------


class TestReroot:
    def test_reroot_to_t1(self) -> None:
        td = _path_decomposition()
        result = compute_reroot(RerootRequest(decomposition=td, root="t1"))
        assert result.root == "t1"
        assert result.parent["t1"] is None
        assert result.children["t1"] == ("t0",)
        assert result.depth["t1"] == 0
        assert result.depth["t0"] == 1
        assert result.paths["t0"] == ["t1", "t0"]

    def test_reroot_preserves_unrooted_tree(self) -> None:
        td = _path_decomposition()
        # Rerooting does not change the width.
        result = compute_reroot(RerootRequest(decomposition=td, root="t0"))
        assert result.parent["t0"] is None
        assert result.children["t0"] == ("t1",)


# ---------------------------------------------------------------------------
# Restrict
# ---------------------------------------------------------------------------


class TestRestrict:
    def test_restrict_to_ab(self) -> None:
        td = _path_decomposition()
        result = compute_restrict(RestrictRequest(decomposition=td, subset=("a", "b")))
        # The restricted graph has vertices {a, b} and edge {(a,b)}.
        assert cast(Any, result.graph["vertices"]) == ("a", "b")
        assert cast(Any, result.graph["edges"]) == (("a", "b"),)
        # The bag {b,c} restricted to {a,b} becomes {b}; bag {a,b} restricted to
        # {a,b} becomes {a,b}. The redundant single-element bag {b} should be
        # pruned if it is contained in its neighbor {a,b}.


# ---------------------------------------------------------------------------
# Bag intersection graph
# ---------------------------------------------------------------------------


class TestBagIntersectionGraph:
    def test_path_bag_intersection_graph(self) -> None:
        td = _path_decomposition()
        result = compute_bag_intersection_graph(
            BagIntersectionGraphRequest(decomposition=td)
        )
        assert len(result.nodes) == 2
        for node in result.nodes:
            assert node["bag_size"] == 2
        assert result.max_adhesion == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_tree_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tree"):
            TreeDecomposition(
                graph=_path_graph(),
                tree_nodes=("t0", "t1"),
                tree_edges=(),
                bags=(("a", "b"), ("b", "c")),
            )

    def test_unsorted_bag_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sorted"):
            TreeDecomposition(
                graph=_path_graph(),
                tree_nodes=("t0",),
                tree_edges=(),
                bags=(("b", "a"),),
            )

    def test_vertex_coverage_rejected(self) -> None:
        # Missing vertex c from all bags.
        with pytest.raises(ValidationError, match="vertex"):
            TreeDecomposition(
                graph=_path_graph(),
                tree_nodes=("t0",),
                tree_edges=(),
                bags=(("a", "b"),),
            )

    def test_edge_coverage_rejected(self) -> None:
        # Graph has two disjoint edges a-b and c-d; vertex d is only in t1
        # and c is only in t0, so edge (c,d) has no single covering bag.
        graph = SimpleUndirectedGraph(
            graph_schema_version="1",
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("c", "d")),
        )
        with pytest.raises(ValidationError, match="edge"):
            TreeDecomposition(
                graph=graph,
                tree_nodes=("t0", "t1"),
                tree_edges=(("t0", "t1"),),
                bags=(("a", "b", "c"), ("a", "b", "d")),
            )

    def test_connectedness_rejected(self) -> None:
        # Vertex a is in t0 and t2 but not in t1, so the containing nodes
        # {t0, t2} are not connected in the path t0-t1-t2.
        graph = SimpleUndirectedGraph(
            graph_schema_version="1",
            vertices=("a",),
            edges=(),
        )
        with pytest.raises(ValidationError, match="connectedness"):
            TreeDecomposition(
                graph=graph,
                tree_nodes=("t0", "t1", "t2"),
                tree_edges=(("t0", "t1"), ("t1", "t2")),
                bags=(("a",), (), ("a",)),
            )

    def test_undeclared_vertex_rejected(self) -> None:
        with pytest.raises(ValidationError, match="undeclared graph vertex"):
            TreeDecomposition(
                graph=_path_graph(),
                tree_nodes=("t0",),
                tree_edges=(),
                bags=(("a", "d"),),
            )
