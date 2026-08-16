"""Tests for the bounded graph isomorphism decision operation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.graph_isomorphism import (
    GraphIsomorphismRequest,
    GraphIsomorphismResult,
)
from jacobian.domains.graph_isomorphism.operations import (
    compute_isomorphism_decision,
)
from jacobian.domains.graph_isomorphism.math_tools import (
    GRAPH_ISOMORPHISM_OPERATIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Empty graph on 3 isolated vertices.
EMPTY_3 = {"vertex_count": 3, "edges": []}

# Path graph P3.
P3_A = {"vertex_count": 3, "edges": [[0, 1], [1, 2]]}
# Same path P3 with relabelled vertices.
P3_B = {"vertex_count": 3, "edges": [[0, 2], [2, 1]]}

# Path graph P4.
P4_A = {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]}
P4_B = {"vertex_count": 4, "edges": [[0, 3], [3, 1], [1, 2]]}

# Cycle graph C4.
C4_A = {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3], [3, 0]]}
C4_B = {"vertex_count": 4, "edges": [[0, 1], [1, 3], [3, 2], [2, 0]]}

# Complete graph K3.
K3_A = {"vertex_count": 3, "edges": [[0, 1], [0, 2], [1, 2]]}
K3_B = {"vertex_count": 3, "edges": [[0, 2], [1, 2], [0, 1]]}

# Complete graph K4.
K4_A = {
    "vertex_count": 4,
    "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
}

# C6: a single 6-cycle (degree sequence all 2).
C6 = {
    "vertex_count": 6,
    "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
}

# 2 x K3: two disjoint triangles (degree sequence all 2, not isomorphic to C6).
TWO_K3 = {
    "vertex_count": 6,
    "edges": [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decide(graph_a: dict, graph_b: dict) -> GraphIsomorphismResult:
    request = GraphIsomorphismRequest.model_validate(
        {"graph_a": graph_a, "graph_b": graph_b}
    )
    return compute_isomorphism_decision(request)


def _assert_valid_mapping(
    result: GraphIsomorphismResult,
    vertex_count: int,
) -> None:
    """Assert the mapping is a valid bijection covering every vertex."""
    assert result.mapping is not None
    sources = [pair[0] for pair in result.mapping]
    targets = [pair[1] for pair in result.mapping]
    assert sorted(sources) == list(range(vertex_count))
    assert sorted(targets) == list(range(vertex_count))


# ---------------------------------------------------------------------------
# Decision tests
# ---------------------------------------------------------------------------

class TestIsomorphismDecision:
    def test_empty_graphs_are_isomorphic(self) -> None:
        result = _decide(EMPTY_3, EMPTY_3)
        assert result.decision == "ISOMORPHIC"
        assert result.mapping is not None
        _assert_valid_mapping(result, 3)

    def test_path_graph_isomorphic(self) -> None:
        result = _decide(P3_A, P3_B)
        assert result.decision == "ISOMORPHIC"
        assert result.mapping is not None
        _assert_valid_mapping(result, 3)

    def test_path_p4_isomorphic(self) -> None:
        result = _decide(P4_A, P4_B)
        assert result.decision == "ISOMORPHIC"
        assert result.mapping is not None
        _assert_valid_mapping(result, 4)

    def test_cycle_c4_isomorphic(self) -> None:
        result = _decide(C4_A, C4_B)
        assert result.decision == "ISOMORPHIC"
        assert result.mapping is not None
        _assert_valid_mapping(result, 4)

    def test_complete_k3_isomorphic(self) -> None:
        result = _decide(K3_A, K3_B)
        assert result.decision == "ISOMORPHIC"
        assert result.mapping is not None
        _assert_valid_mapping(result, 3)

    def test_complete_k4_isomorphic(self) -> None:
        result = _decide(K4_A, K4_A)
        assert result.decision == "ISOMORPHIC"
        assert result.mapping is not None
        _assert_valid_mapping(result, 4)


# ---------------------------------------------------------------------------
# Non-isomorphic cases
# ---------------------------------------------------------------------------

class TestNonIsomorphic:
    def test_empty_vs_path_not_isomorphic(self) -> None:
        result = _decide(EMPTY_3, P3_A)
        assert result.decision == "NOT_ISOMORPHIC"
        assert result.mapping is None

    def test_path_vs_cycle_not_isomorphic(self) -> None:
        # P4 vs C4 — same vertex count, different edge counts.
        result = _decide(P4_A, C4_A)
        assert result.decision == "NOT_ISOMORPHIC"
        assert result.mapping is None

    def test_same_degree_sequence_non_isomorphic(self) -> None:
        # C6 (one 6-cycle) vs 2 x K3 (two disjoint triangles) both have
        # degree sequence (2, 2, 2, 2, 2, 2) but are not isomorphic.
        result = _decide(C6, TWO_K3)
        assert result.decision == "NOT_ISOMORPHIC"
        assert result.mapping is None


# ---------------------------------------------------------------------------
# Certificate correctness
# ---------------------------------------------------------------------------

class TestCertificateCorrectness:
    def test_mapping_preserves_edges(self) -> None:
        """The mapped vertices must preserve adjacency of the source graph."""
        result = _decide(P4_A, P4_B)
        assert result.decision == "ISOMORPHIC"
        assert result.mapping is not None

        mapping = dict(result.mapping)
        for u, v in P4_A["edges"]:
            mapped_u, mapped_v = mapping[u], mapping[v]
            expected = tuple(sorted((mapped_u, mapped_v)))
            assert expected in P4_B["edges"] or tuple(
                sorted((mapped_u, mapped_v))
            ) in [tuple(sorted(e)) for e in P4_B["edges"]]

    def test_mapping_is_a_bijection(self) -> None:
        result = _decide(C4_A, C4_B)
        assert result.decision == "ISOMORPHIC"
        _assert_valid_mapping(result, 4)


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_mismatched_vertex_counts_rejected(self) -> None:
        with pytest.raises(ValidationError, match="same vertex count"):
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": P3_A,
                    "graph_b": {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]},
                }
            )

    def test_self_loop_rejected(self) -> None:
        with pytest.raises(ValidationError, match="self-loops"):
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {"vertex_count": 3, "edges": [[0, 0], [1, 2]]},
                    "graph_b": P3_A,
                }
            )

    def test_duplicate_edge_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate edges"):
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 0], [1, 2]],
                    },
                    "graph_b": P3_A,
                }
            )

    def test_out_of_range_vertex_rejected(self) -> None:
        with pytest.raises(ValidationError, match="edge vertices must be in"):
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {"vertex_count": 3, "edges": [[0, 3]]},
                    "graph_b": P3_A,
                }
            )


# ---------------------------------------------------------------------------
# Math tool registration and discovery
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_operation_is_registered(self) -> None:
        op_ids = [op.operation_id for op in GRAPH_ISOMORPHISM_OPERATIONS]
        assert "graph.isomorphism.decide" in op_ids

    def test_operation_has_tags(self) -> None:
        op = GRAPH_ISOMORPHISM_OPERATIONS[0]
        assert "graph" in op.tags
        assert "isomorphism" in op.tags
        assert "exact" in op.tags
