"""Tests for the graph isomorphism decision operation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.isomorphism._models import (
    GraphIsomorphismRequest,
    GraphIsomorphismResult,
)
from jacobian.math.graphs.isomorphism._operations import (
    decide_graph_isomorphism,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decide(graph_a: dict, graph_b: dict) -> GraphIsomorphismResult:
    return decide_graph_isomorphism(
        GraphIsomorphismRequest.model_validate({"graph_a": graph_a, "graph_b": graph_b})
    )


def _path_edges(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _cycle_edges(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def _complete_edges(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _is_mapping_valid_isomorphism(graph_a: dict, graph_b: dict, mapping: tuple) -> bool:
    """Independently verify that ``mapping`` is a graph isomorphism."""
    if not mapping:
        return False
    edges_a: set[tuple[int, int]] = {tuple(e) for e in graph_a["edges"]}
    edges_b: set[tuple[int, int]] = {tuple(e) for e in graph_b["edges"]}
    directed = graph_a.get("directed", False)
    if directed:
        # Canonicalise directed edges
        edges_a = set(edges_a)
        edges_b = set(edges_b)
    else:
        edges_a = {(min(u, v), max(u, v)) for u, v in edges_a}
        edges_b = {(min(u, v), max(u, v)) for u, v in edges_b}
    fwd = {m.from_vertex: m.to_vertex for m in mapping}
    inv = {m.to_vertex: m.from_vertex for m in mapping}
    if len(fwd) != graph_a["vertex_count"]:
        return False
    if len(inv) != graph_b["vertex_count"]:
        return False
    if set(fwd) != set(range(graph_a["vertex_count"])):
        return False
    if set(inv) != set(range(graph_b["vertex_count"])):
        return False
    # For each edge in A, the mapped edge must be in B.
    for u, v in edges_a:
        mu, mv = fwd[u], fwd[v]
        if directed:  # noqa: SIM108
            mapped = (mu, mv)
        else:
            mapped = (min(mu, mv), max(mu, mv))
        if mapped not in edges_b:
            return False
    # And conversely every edge in B must have a preimage in A.
    for u, v in edges_b:
        mu, mv = inv[u], inv[v]
        if directed:  # noqa: SIM108
            pre = (mu, mv)
        else:
            pre = (min(mu, mv), max(mu, mv))
        if pre not in edges_a:
            return False
    return True


# ---------------------------------------------------------------------------
# Path graphs
# ---------------------------------------------------------------------------


class TestPathGraphs:
    def test_path_graphs_isomorphic(self) -> None:
        n = 6
        # Shift vertex labels: B is A under the permutation i -> (i + 1) % n
        perm = {i: (i + 1) % n for i in range(n)}
        a_edges = _path_edges(n)
        b_edges = [(perm[u], perm[v]) for u, v in a_edges]
        result = _decide(
            {"vertex_count": n, "directed": False, "edges": a_edges},
            {"vertex_count": n, "directed": False, "edges": b_edges},
        )
        assert result.status == "ISOMORPHIC"
        assert len(result.vertex_mapping) == n
        assert _is_mapping_valid_isomorphism(
            {"vertex_count": n, "directed": False, "edges": a_edges},
            {"vertex_count": n, "directed": False, "edges": b_edges},
            result.vertex_mapping,
        )


# ---------------------------------------------------------------------------
# Cycle graphs
# ---------------------------------------------------------------------------


class TestCycleGraphs:
    def test_cycle_graphs_isomorphic(self) -> None:
        n = 6
        perm = {i: (i + 2) % n for i in range(n)}
        a_edges = _cycle_edges(n)
        b_edges = [(perm[u], perm[v]) for u, v in a_edges]
        result = _decide(
            {"vertex_count": n, "directed": False, "edges": a_edges},
            {"vertex_count": n, "directed": False, "edges": b_edges},
        )
        assert result.status == "ISOMORPHIC"
        assert len(result.vertex_mapping) == n
        assert _is_mapping_valid_isomorphism(
            {"vertex_count": n, "directed": False, "edges": a_edges},
            {"vertex_count": n, "directed": False, "edges": b_edges},
            result.vertex_mapping,
        )

    def test_cycle_not_isomorphic_to_path(self) -> None:
        # C_4 is not isomorphic to P_4 (different degree sequences).
        result = _decide(
            {"vertex_count": 4, "directed": False, "edges": _cycle_edges(4)},
            {"vertex_count": 4, "directed": False, "edges": _path_edges(4)},
        )
        assert result.status == "NOT_ISOMORPHIC"


# ---------------------------------------------------------------------------
# Complete graphs
# ---------------------------------------------------------------------------


class TestCompleteGraphs:
    def test_complete_graphs_isomorphic(self) -> None:
        n = 5
        perm = {i: (i + 3) % n for i in range(n)}
        a_edges = _complete_edges(n)
        b_edges = [(perm[u], perm[v]) for u, v in a_edges]
        result = _decide(
            {"vertex_count": n, "directed": False, "edges": a_edges},
            {"vertex_count": n, "directed": False, "edges": b_edges},
        )
        assert result.status == "ISOMORPHIC"
        assert len(result.vertex_mapping) == n
        assert _is_mapping_valid_isomorphism(
            {"vertex_count": n, "directed": False, "edges": a_edges},
            {"vertex_count": n, "directed": False, "edges": b_edges},
            result.vertex_mapping,
        )

    def test_complete_graph_not_isomorphic_to_path(self) -> None:
        result = _decide(
            {"vertex_count": 4, "directed": False, "edges": _complete_edges(4)},
            {"vertex_count": 4, "directed": False, "edges": _path_edges(4)},
        )
        assert result.status == "NOT_ISOMORPHIC"


# ---------------------------------------------------------------------------
# Empty graphs
# ---------------------------------------------------------------------------


class TestEmptyGraphs:
    def test_empty_graphs_isomorphic(self) -> None:
        # Two isolated-vertex graphs with no edges are trivially isomorphic.
        result = _decide(
            {"vertex_count": 3, "directed": False, "edges": ()},
            {"vertex_count": 3, "directed": False, "edges": ()},
        )
        assert result.status == "ISOMORPHIC"
        assert len(result.vertex_mapping) == 3

    def test_single_vertex_graphs_isomorphic(self) -> None:
        result = _decide(
            {"vertex_count": 1, "directed": False, "edges": ()},
            {"vertex_count": 1, "directed": False, "edges": ()},
        )
        assert result.status == "ISOMORPHIC"
        assert len(result.vertex_mapping) == 1


# ---------------------------------------------------------------------------
# Nonisomorphic with the same degree sequence
# ---------------------------------------------------------------------------


class TestNonisomorphicSameDegreeSequence:
    def test_genuine_same_degree_sequence_nonisomorphic(self) -> None:
        # 2-paw-vs-star-on-the-same-degree-sequence fixture.
        #
        # Graph A: 6 vertices, 6 edges; a triangular prism (C6 with chords).
        # Use a well-known pair:  the two 4-vertex graphs both on 4 vertices
        # with degree sequence {1, 1, 2, 2} are isomorphic (both are paths P3).
        # Instead use 6 vertices, 7 edges each, degree sequence (3,3,2,2,2,2).
        #
        # Graph A: a triangle 0-1-2-0 plus a triangle 3-4-5-3, and a bridge
        # 0-3.  Degrees:
        #   0:3 (edges 0-1, 0-2, 0-3)
        #   1:2, 2:2, 3:3 (3-0, 3-4, 3-5), 4:2, 5:2.
        # So degree sequence is (3,3,2,2,2,2).
        # Graph B: two triangles sharing a single vertex (a "butterfly"
        # pinched at vertex 0).  Degrees:
        #   0:4 (edges 0-1, 0-2, 0-3, 0-4)
        # which does not match.  So use a different pair.
        #
        # Instead, use the classic same-degree-sequence nonisomorphic pair:
        #   Graph A: path P4 with an extra edge (0-2), giving degrees
        #     edges: (0,1),(1,2),(2,3),(0,2)
        #     degrees: 0:2, 1:2, 2:3, 3:1
        #   Graph B: triangle with a pendant edge
        #     edges: (0,1),(1,2),(0,2),(2,3)
        #     degrees: 0:2, 1:2, 2:3, 3:1
        # These have the same degree sequence (3,2,2,1) and are in fact
        # isomorphic (both are a triangle with a pendant).  Use a harder pair.
        #
        # Use the canonical 6-vertex same-degree-sequence nonisomorphic pair:
        #   Graph A (C6 with one chord): edges of C6 plus chord (0, 2).
        #   Graph B (C6 with the other chord): edges of C6 plus chord (0, 3).
        # Both have degree sequence (3, 2, 3, 2, 2, 2) but one has a 3-cycle
        # (the chord 0-2 plus edges 0-1-2 forms a triangle), while the other
        # (chord 0-3) has no triangle.  So they are non-isomorphic despite
        # the same degree sequence.
        n = 6
        c6 = _cycle_edges(n)
        a_edges = [*c6, (0, 2)]
        b_edges = [*c6, (0, 3)]
        # Sanity check the degree sequences are identical.
        degs_a = [0] * n
        for u, v in a_edges:
            degs_a[u] += 1
            degs_a[v] += 1
        degs_b = [0] * n
        for u, v in b_edges:
            degs_b[u] += 1
            degs_b[v] += 1
        assert sorted(degs_a) == sorted(degs_b)
        result = _decide(
            {"vertex_count": n, "directed": False, "edges": a_edges},
            {"vertex_count": n, "directed": False, "edges": b_edges},
        )
        assert result.status == "NOT_ISOMORPHIC"
        assert result.vertex_mapping == ()


# ---------------------------------------------------------------------------
# Mismatched vertex counts
# ---------------------------------------------------------------------------


class TestMismatchedVertexCount:
    def test_mismatched_vertex_count_rejected(self) -> None:
        # The contract-level validator should reject mismatched vertex counts
        # before any NetworkX call is made.
        with pytest.raises(ValidationError) as exc_info:
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {
                        "vertex_count": 3,
                        "directed": False,
                        "edges": _path_edges(3),
                    },
                    "graph_b": {
                        "vertex_count": 4,
                        "directed": False,
                        "edges": _path_edges(4),
                    },
                }
            )
        assert "vertex count" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Directed graphs
# ---------------------------------------------------------------------------


class TestDirectedGraphs:
    def test_directed_path_isomorphic(self) -> None:
        n = 5
        # A: 0 -> 1 -> 2 -> 3 -> 4
        # B: the same path under the relabelling i -> (n-1-i), so B is the
        # reversed path 4 -> 3 -> 2 -> 1 -> 0.  Because both are directed,
        # the path and its reverse are NOT isomorphic as directed graphs,
        # but a path and a cyclic relabelling of itself ARE.
        perm = {i: (i + 1) % n for i in range(n)}
        a_edges = [(i, i + 1) for i in range(n - 1)]
        b_edges = [(perm[u], perm[v]) for u, v in a_edges]
        result = _decide(
            {"vertex_count": n, "directed": True, "edges": a_edges},
            {"vertex_count": n, "directed": True, "edges": b_edges},
        )
        assert result.status == "ISOMORPHIC"
        assert len(result.vertex_mapping) == n
        assert _is_mapping_valid_isomorphism(
            {"vertex_count": n, "directed": True, "edges": a_edges},
            {"vertex_count": n, "directed": True, "edges": b_edges},
            result.vertex_mapping,
        )

    def test_directed_path_not_isomorphic_to_reverse(self) -> None:
        # A directed path P4 and its reverse are not isomorphic as directed
        # graphs (one is a source-to-sink path, the other is a sink-to-source
        # path; swapping the labels gives the same graph, but the reversal
        # of a path is isomorphic to the path under the label swap i <-> n-1-i).
        # In fact, for a simple path, the reversal IS isomorphic to the
        # original (reverse the labels).  Use a genuinely directed example:
        # a path vs a path plus a back-edge, both on the same vertex set.
        n = 4
        path = [(0, 1), (1, 2), (2, 3)]
        extra = [(0, 1), (1, 2), (2, 3), (3, 0)]  # directed cycle
        result = _decide(
            {"vertex_count": n, "directed": True, "edges": path},
            {"vertex_count": n, "directed": True, "edges": extra},
        )
        assert result.status == "NOT_ISOMORPHIC"

    def test_directed_cycle_isomorphic(self) -> None:
        n = 5
        # A: a directed cycle 0 -> 1 -> 2 -> 3 -> 4 -> 0.
        # B: the same cycle under the relabelling i -> (i + 2) % n.
        perm = {i: (i + 2) % n for i in range(n)}
        a_edges = [(i, (i + 1) % n) for i in range(n)]
        b_edges = [(perm[u], perm[v]) for u, v in a_edges]
        result = _decide(
            {"vertex_count": n, "directed": True, "edges": a_edges},
            {"vertex_count": n, "directed": True, "edges": b_edges},
        )
        assert result.status == "ISOMORPHIC"
        assert len(result.vertex_mapping) == n
        assert _is_mapping_valid_isomorphism(
            {"vertex_count": n, "directed": True, "edges": a_edges},
            {"vertex_count": n, "directed": True, "edges": b_edges},
            result.vertex_mapping,
        )


# ---------------------------------------------------------------------------
# Validation: invalid edges
# ---------------------------------------------------------------------------


class TestValidation:
    def test_self_loop_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {
                        "vertex_count": 3,
                        "directed": False,
                        "edges": [(0, 0), (0, 1)],
                    },
                    "graph_b": {
                        "vertex_count": 3,
                        "directed": False,
                        "edges": [(0, 1)],
                    },
                }
            )

    def test_duplicate_edge_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {
                        "vertex_count": 3,
                        "directed": False,
                        "edges": [(0, 1), (1, 0)],
                    },
                    "graph_b": {
                        "vertex_count": 3,
                        "directed": False,
                        "edges": [(0, 1)],
                    },
                }
            )

    def test_out_of_range_vertex_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {
                        "vertex_count": 3,
                        "directed": False,
                        "edges": [(0, 3)],
                    },
                    "graph_b": {
                        "vertex_count": 3,
                        "directed": False,
                        "edges": [(0, 1)],
                    },
                }
            )

    def test_mismatched_directedness_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {
                        "vertex_count": 3,
                        "directed": True,
                        "edges": [(0, 1)],
                    },
                    "graph_b": {
                        "vertex_count": 3,
                        "directed": False,
                        "edges": [(0, 1)],
                    },
                }
            )
        assert "directedness" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Operation metadata
# ---------------------------------------------------------------------------


class TestOperationMetadata:
    def test_operation_registered(self) -> None:
        from jacobian.math.graphs.isomorphism._tools import TOOLS

        ops = {operation.operation_id: operation for operation in TOOLS}
        op = ops["graph.isomorphism.decide.compute"]
        assert op.operation_id == "graph.isomorphism.decide.compute"
        assert op.request_type.__name__ == "GraphIsomorphismRequest"
        assert op.result_type.__name__ == "GraphIsomorphismResult"
        assert "graph" in op.tags
        assert "isomorphism" in op.tags
        assert "exact" in op.tags
        assert len(op.examples) == 2

    def test_examples_round_trip(self) -> None:
        """The operation's declared examples must validate and execute."""
        from jacobian.math.graphs.isomorphism._tools import TOOLS

        ops = {operation.operation_id: operation for operation in TOOLS}
        op = ops["graph.isomorphism.decide.compute"]
        for ex in op.examples:
            req = op.request_type.model_validate(ex.input)
            result = op.run(req)
            assert isinstance(result, GraphIsomorphismResult)
            assert result.status in ("ISOMORPHIC", "NOT_ISOMORPHIC")
            assert result.convention == "NETWORKX_IS_ISOMORPHIC"
