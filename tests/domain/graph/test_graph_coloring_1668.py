"""Tests for bounded exact graph coloring and independent set operations (#1668).

Covers the acceptance criteria:
- 3-colorable graph (triangle K3, bipartite graph)
- non-3-colorable graph (K4)
- maximum independent set correctness
- maximal independent set decision (MAXIMAL, NOT_INDEPENDENT, INDEPENDENT_NOT_MAXIMAL)
- fail-closed on unsupported graph types (self-loops, duplicate edges, out-of-range vertices)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.graph_coloring_ops import (
    KColorabilityRequest,
    MaximalIndependentSetRequest,
    MaximumIndependentSetRequest,
)
from jacobian.domains.graph_coloring_ops.operations import (
    compute_k_colorability,
    compute_maximal_independent_set_decision,
    compute_maximum_independent_set,
)


# ---------------------------------------------------------------------------
# k-colorability
# ---------------------------------------------------------------------------


def test_triangle_is_3_colorable() -> None:
    """K3 is 3-colorable but not 2-colorable."""
    request = KColorabilityRequest.model_validate(
        {
            "graph": {"vertex_count": 3, "edges": [[0, 1], [1, 2], [2, 0]]},
            "colors": 3,
        }
    )
    result = compute_k_colorability(request)
    assert result.colorable is True
    assert result.coloring is not None
    # Verify proper coloring: adjacent vertices have distinct colors.
    for u, v in request.graph.edges:
        assert result.coloring[u] != result.coloring[v]
    # All colors are in range [0, k).
    assert all(0 <= c < 3 for c in result.coloring)


def test_triangle_is_not_2_colorable() -> None:
    """K3 is not 2-colorable (odd cycle)."""
    request = KColorabilityRequest.model_validate(
        {
            "graph": {"vertex_count": 3, "edges": [[0, 1], [1, 2], [2, 0]]},
            "colors": 2,
        }
    )
    result = compute_k_colorability(request)
    assert result.colorable is False
    assert result.coloring is None


def test_k4_is_not_3_colorable() -> None:
    """K4 (complete graph on 4 vertices) is not 3-colorable."""
    request = KColorabilityRequest.model_validate(
        {
            "graph": {
                "vertex_count": 4,
                "edges": [
                    [0, 1], [0, 2], [0, 3],
                    [1, 2], [1, 3], [2, 3],
                ],
            },
            "colors": 3,
        }
    )
    result = compute_k_colorability(request)
    assert result.colorable is False
    assert result.coloring is None


def test_k4_is_4_colorable() -> None:
    """K4 is 4-colorable."""
    request = KColorabilityRequest.model_validate(
        {
            "graph": {
                "vertex_count": 4,
                "edges": [
                    [0, 1], [0, 2], [0, 3],
                    [1, 2], [1, 3], [2, 3],
                ],
            },
            "colors": 4,
        }
    )
    result = compute_k_colorability(request)
    assert result.colorable is True
    assert result.coloring is not None
    for u, v in request.graph.edges:
        assert result.coloring[u] != result.coloring[v]


def test_bipartite_graph_is_2_colorable() -> None:
    """A bipartite graph (K3,3) is 2-colorable."""
    request = KColorabilityRequest.model_validate(
        {
            "graph": {
                "vertex_count": 6,
                "edges": [
                    [0, 3], [0, 4], [0, 5],
                    [1, 3], [1, 4], [1, 5],
                    [2, 3], [2, 4], [2, 5],
                ],
            },
            "colors": 2,
        }
    )
    result = compute_k_colorability(request)
    assert result.colorable is True
    assert result.coloring is not None
    for u, v in request.graph.edges:
        assert result.coloring[u] != result.coloring[v]


def test_bipartite_graph_is_not_3_uncolorable() -> None:
    """A bipartite graph is also 3-colorable (more colors = easier)."""
    request = KColorabilityRequest.model_validate(
        {
            "graph": {
                "vertex_count": 6,
                "edges": [
                    [0, 3], [0, 4], [0, 5],
                    [1, 3], [1, 4], [1, 5],
                    [2, 3], [2, 4], [2, 5],
                ],
            },
            "colors": 3,
        }
    )
    result = compute_k_colorability(request)
    assert result.colorable is True
    assert result.coloring is not None


def test_path_graph_is_2_colorable() -> None:
    """A path graph (tree) is always 2-colorable."""
    request = KColorabilityRequest.model_validate(
        {
            "graph": {"vertex_count": 5, "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]},
            "colors": 2,
        }
    )
    result = compute_k_colorability(request)
    assert result.colorable is True
    assert result.coloring is not None
    for u, v in request.graph.edges:
        assert result.coloring[u] != result.coloring[v]


def test_single_vertex_is_1_colorable() -> None:
    """A single vertex is 1-colorable."""
    request = KColorabilityRequest.model_validate(
        {"graph": {"vertex_count": 1, "edges": []}, "colors": 1}
    )
    result = compute_k_colorability(request)
    assert result.colorable is True
    assert result.coloring is not None
    assert result.coloring == (0,)


# ---------------------------------------------------------------------------
# Maximum independent set
# ---------------------------------------------------------------------------


def test_maximum_independent_set_of_path() -> None:
    """Maximum independent set of P4 (path on 4 vertices) has cardinality 2."""
    request = MaximumIndependentSetRequest.model_validate(
        {"graph": {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]}}
    )
    result = compute_maximum_independent_set(request)
    assert result.cardinality == 2
    # Verify independence: no edge has both endpoints in the set.
    for u, v in request.graph.edges:
        assert not (u in result.independent_set and v in result.independent_set)


def test_maximum_independent_set_of_complete_graph() -> None:
    """Maximum independent set of K4 is 1 (only one vertex can be chosen)."""
    request = MaximumIndependentSetRequest.model_validate(
        {
            "graph": {
                "vertex_count": 4,
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            }
        }
    )
    result = compute_maximum_independent_set(request)
    assert result.cardinality == 1
    assert len(result.independent_set) == 1


def test_maximum_independent_set_of_empty_graph() -> None:
    """Maximum independent set of an empty graph is all vertices."""
    request = MaximumIndependentSetRequest.model_validate(
        {"graph": {"vertex_count": 5, "edges": []}}
    )
    result = compute_maximum_independent_set(request)
    assert result.cardinality == 5
    assert set(result.independent_set) == {0, 1, 2, 3, 4}


def test_maximum_independent_set_of_cycle() -> None:
    """Maximum independent set of C5 has cardinality 2."""
    request = MaximumIndependentSetRequest.model_validate(
        {
            "graph": {
                "vertex_count": 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            }
        }
    )
    result = compute_maximum_independent_set(request)
    assert result.cardinality == 2
    for u, v in request.graph.edges:
        assert not (u in result.independent_set and v in result.independent_set)


# ---------------------------------------------------------------------------
# Maximal independent set decision
# ---------------------------------------------------------------------------


def test_maximal_independent_set_maximal() -> None:
    """{0, 2} is a maximal independent set of P4."""
    request = MaximalIndependentSetRequest.model_validate(
        {
            "graph": {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]},
            "candidate_set": [0, 2],
        }
    )
    result = compute_maximal_independent_set_decision(request)
    assert result.decision == "MAXIMAL"


def test_maximal_independent_set_not_independent() -> None:
    """{0, 1} is not an independent set of P4 (edge 0-1)."""
    request = MaximalIndependentSetRequest.model_validate(
        {
            "graph": {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]},
            "candidate_set": [0, 1],
        }
    )
    result = compute_maximal_independent_set_decision(request)
    assert result.decision == "NOT_INDEPENDENT"


def test_maximal_independent_set_independent_not_maximal() -> None:
    """{0} is independent but not maximal in P4 (vertex 2 can be added)."""
    request = MaximalIndependentSetRequest.model_validate(
        {
            "graph": {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]},
            "candidate_set": [0],
        }
    )
    result = compute_maximal_independent_set_decision(request)
    assert result.decision == "INDEPENDENT_NOT_MAXIMAL"


def test_maximal_independent_set_empty_is_not_maximal() -> None:
    """The empty set is independent but not maximal if the graph has vertices."""
    request = MaximalIndependentSetRequest.model_validate(
        {
            "graph": {"vertex_count": 3, "edges": [[0, 1]]},
            "candidate_set": [],
        }
    )
    result = compute_maximal_independent_set_decision(request)
    assert result.decision == "INDEPENDENT_NOT_MAXIMAL"


def test_maximal_independent_set_empty_is_maximal_in_empty_graph() -> None:
    """The empty set is maximal in a graph with no vertices — but we need at least 1 vertex.

    Actually, with vertex_count >= 1 and no edges, the empty set is NOT maximal
    because any vertex can be added. Let us test the single vertex case: {0} is maximal.
    """
    request = MaximalIndependentSetRequest.model_validate(
        {
            "graph": {"vertex_count": 1, "edges": []},
            "candidate_set": [0],
        }
    )
    result = compute_maximal_independent_set_decision(request)
    assert result.decision == "MAXIMAL"


def test_maximal_independent_set_complete_graph() -> None:
    """In K3, {0} is a maximal independent set (singletons are maximal in K_n)."""
    request = MaximalIndependentSetRequest.model_validate(
        {
            "graph": {"vertex_count": 3, "edges": [[0, 1], [1, 2], [0, 2]]},
            "candidate_set": [0],
        }
    )
    result = compute_maximal_independent_set_decision(request)
    assert result.decision == "MAXIMAL"


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def test_self_loop_rejected() -> None:
    """Self-loops are rejected by the graph contract."""
    with pytest.raises(ValidationError, match="self-loops"):
        KColorabilityRequest.model_validate(
            {"graph": {"vertex_count": 3, "edges": [[0, 0], [0, 1]]}, "colors": 2}
        )


def test_duplicate_edges_rejected() -> None:
    """Duplicate edges are rejected."""
    with pytest.raises(ValidationError, match="duplicate"):
        KColorabilityRequest.model_validate(
            {
                "graph": {"vertex_count": 3, "edges": [[0, 1], [1, 0]]},
                "colors": 2,
            }
        )


def test_out_of_range_vertex_rejected() -> None:
    """Vertices outside [0, vertex_count) are rejected."""
    with pytest.raises(ValidationError, match="edge vertices must be"):
        KColorabilityRequest.model_validate(
            {"graph": {"vertex_count": 3, "edges": [[0, 5]]}, "colors": 2}
        )


def test_candidate_out_of_range_rejected() -> None:
    """Candidate vertices outside [0, vertex_count) are rejected."""
    with pytest.raises(ValidationError, match="candidate vertices must lie"):
        MaximalIndependentSetRequest.model_validate(
            {
                "graph": {"vertex_count": 3, "edges": [[0, 1]]},
                "candidate_set": [0, 5],
            }
        )


def test_duplicate_candidate_rejected() -> None:
    """Duplicate candidate vertices are rejected."""
    with pytest.raises(ValidationError, match="duplicate"):
        MaximalIndependentSetRequest.model_validate(
            {
                "graph": {"vertex_count": 3, "edges": [[0, 1]]},
                "candidate_set": [0, 0],
            }
        )


def test_vertex_count_upper_bound_enforced() -> None:
    """vertex_count > 20 is rejected (bounded operation)."""
    with pytest.raises(ValidationError):
        KColorabilityRequest.model_validate(
            {"graph": {"vertex_count": 21, "edges": []}, "colors": 2}
        )
