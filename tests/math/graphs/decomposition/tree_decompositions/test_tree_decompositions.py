"""Tests for tree-decomposition operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.catalog.models import OperationResult
from jacobian.math.graphs.decomposition.tree_decompositions import TreeDecomposition
from jacobian.math.graphs.decomposition.tree_decompositions._models import (
    AdhesionsRequest,
    BagIntersectionGraphRequest,
    RerootRequest,
    RestrictRequest,
    VertexOccurrencesRequest,
    WidthRequest,
)
from jacobian.math.graphs.decomposition.tree_decompositions._operations import (
    compute_adhesions,
    compute_bag_intersection_graph,
    compute_reroot,
    compute_restrict,
    compute_vertex_occurrences,
    compute_width,
)
from jacobian.math.graphs.decomposition.tree_decompositions._tools import TOOLS
from jacobian.math.graphs.values import SimpleUndirectedGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_graph() -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
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


def _labeled_path_decomposition(
    *, node_count: int, label_body: str
) -> TreeDecomposition:
    nodes = tuple(f"n{index:03d}_{label_body}" for index in range(node_count))
    return TreeDecomposition(
        graph=SimpleUndirectedGraph(vertices=("a",), edges=()),
        tree_nodes=nodes,
        tree_edges=tuple(
            (nodes[index], nodes[index + 1]) for index in range(node_count - 1)
        ),
        bags=(("a",),) * node_count,
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
        assert set(per_vertex["b"].nodes) == {"t0", "t1"}
        assert per_vertex["b"].count == 2
        assert per_vertex["b"].edges == (("t0", "t1"),)
        # Vertex a appears in one bag.
        assert per_vertex["a"].nodes == ("t0",)
        assert per_vertex["a"].count == 1
        # Vertex c appears in one bag.
        assert per_vertex["c"].nodes == ("t1",)
        assert per_vertex["c"].count == 1


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
        assert edge.edge == ("t0", "t1")
        assert edge.adhesion == ("b",)
        assert edge.size == 1
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

    def test_admits_decomposed_labels_whose_canonical_projection_fits(self) -> None:
        # Each raw label encodes to 337 bytes, so measuring the unnormalized
        # spelling exceeds the 10 MiB transport limit; NFC composes each
        # e + U+0301 pair into a 2-byte character and the canonical
        # projection of the same request fits.
        td = _labeled_path_decomposition(node_count=256, label_body="e\u0301" * 110)
        request = RerootRequest(decomposition=td, root=td.tree_nodes[0])
        projected = compute_reroot(request).model_dump(mode="json")
        assert len(canonicalize_json(projected)) <= CanonicalLimits().max_output_bytes
        public_result = OperationResult(
            operation_id="graph.tree_decomposition.reroot.compute",
            runtime_ms=0,
            output=projected,
        )
        assert (
            OperationResult.model_validate_json(public_result.model_dump_json())
            == public_result
        )

    def test_rejects_projection_over_the_canonical_transport_limit(self) -> None:
        # ASCII labels are unchanged by NFC, so the canonical projection
        # itself exceeds the limit and admission must still reject.
        td = _labeled_path_decomposition(node_count=256, label_body="x" * 394)
        with pytest.raises(ValidationError) as exc_info:
            RerootRequest(decomposition=td, root=td.tree_nodes[0])
        assert exc_info.value.errors()[0]["type"] == (
            "graph.reroot_result_exceeds_transport_limit"
        )

    def test_rejects_labels_colliding_after_canonicalization(self) -> None:
        # The raw spellings are distinct, so TreeDecomposition admits them;
        # NFC composes both to the same key and delivery would reject the
        # result maps, so admission must reject the request instead.
        nodes = ("e\u0301x", "\u00e9x")
        td = TreeDecomposition(
            graph=SimpleUndirectedGraph(vertices=("a",), edges=()),
            tree_nodes=nodes,
            tree_edges=((nodes[0], nodes[1]),),
            bags=(("a",), ("a",)),
        )
        with pytest.raises(ValidationError) as exc_info:
            RerootRequest(decomposition=td, root=nodes[0])
        assert exc_info.value.errors()[0]["type"] == (
            "graph.reroot_tree_node_labels_collide_after_normalization"
        )


# ---------------------------------------------------------------------------
# Restrict
# ---------------------------------------------------------------------------


class TestRestrict:
    def test_restrict_to_ab(self) -> None:
        td = _path_decomposition()
        result = compute_restrict(RestrictRequest(decomposition=td, subset=("a", "b")))
        # The restricted graph has vertices {a, b} and edge {(a,b)}.
        assert result.graph.vertices == ("a", "b")
        assert result.graph.edges == (("a", "b"),)
        # The bag {b,c} restricted to {a,b} becomes {b}; bag {a,b} restricted to
        # {a,b} becomes {a,b}. The redundant single-element bag {b} should be
        # pruned if it is contained in its neighbor {a,b}.

    def test_rejects_vertices_outside_the_source_graph(self) -> None:
        with pytest.raises(ValidationError):
            RestrictRequest(
                decomposition=_path_decomposition(),
                subset=("a", "missing"),
            )

    def test_prunes_against_the_contracted_tree(self) -> None:
        decomposition = TreeDecomposition(
            graph=SimpleUndirectedGraph(vertices=("a", "b", "c", "d"), edges=()),
            tree_nodes=("left", "center", "deleted", "right"),
            tree_edges=(
                ("left", "center"),
                ("center", "deleted"),
                ("deleted", "right"),
            ),
            bags=(("a", "b"), ("a", "d"), ("d",), ("c", "d")),
        )

        result = compute_restrict(
            RestrictRequest(
                decomposition=decomposition,
                subset=("a", "b", "c"),
            )
        )

        assert result.tree_nodes == ("left", "center", "right")
        assert result.tree_edges == (("center", "left"), ("center", "right"))
        assert result.bags == (("a", "b"), ("a",), ("c",))


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
            assert node.bag_size == 2
        assert result.max_adhesion == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_tree_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TreeDecomposition(
                graph=_path_graph(),
                tree_nodes=("t0", "t1"),
                tree_edges=(),
                bags=(("a", "b"), ("b", "c")),
            )

    def test_unsorted_bag_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TreeDecomposition(
                graph=_path_graph(),
                tree_nodes=("t0",),
                tree_edges=(),
                bags=(("b", "a"),),
            )

    def test_vertex_coverage_rejected(self) -> None:
        # Missing vertex c from all bags.
        with pytest.raises(ValidationError):
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
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("c", "d")),
        )
        with pytest.raises(ValidationError):
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
            vertices=("a",),
            edges=(),
        )
        with pytest.raises(ValidationError):
            TreeDecomposition(
                graph=graph,
                tree_nodes=("t0", "t1", "t2"),
                tree_edges=(("t0", "t1"), ("t1", "t2")),
                bags=(("a",), (), ("a",)),
            )

    def test_undeclared_vertex_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TreeDecomposition(
                graph=_path_graph(),
                tree_nodes=("t0",),
                tree_edges=(),
                bags=(("a", "d"),),
            )
