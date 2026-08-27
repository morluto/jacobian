"""Tests for graph.directed.dag_longest_path.compute."""

from __future__ import annotations

from jacobian.math.graphs.directed._models import (
    DagLongestPathRequest,
    DagLongestPathResult,
)
from jacobian.math.graphs.directed._operations import compute_dag_longest_path


def _longest_path(graph: dict[str, object]) -> DagLongestPathResult:
    return compute_dag_longest_path(
        DagLongestPathRequest.model_validate({"graph": graph})
    )


class TestDagLongestPath:
    def test_empty_edge_dag_max_path_is_zero(self) -> None:
        """An edgeless DAG has a longest path of zero edges."""
        result = _longest_path({"vertex_count": 3, "edges": []})
        assert result.status == "ACYCLIC"
        assert result.maximum_edge_count == 0
        assert len(result.path) == 1

    def test_simple_path_graph(self) -> None:
        """A linear chain 0 -> 1 -> 2 -> 3 has longest path of 3 edges."""
        result = _longest_path({"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]})
        assert result.status == "ACYCLIC"
        assert result.maximum_edge_count == 3
        assert result.path == (0, 1, 2, 3)

    def test_fork_join_tie_lexicographic(self) -> None:
        """Diamond DAG: 0 -> {1, 2} -> 3.

        Both paths 0->1->3 and 0->2->3 have 2 edges.  The lexicographically
        least path witness is (0, 1, 3).
        """
        result = _longest_path(
            {"vertex_count": 4, "edges": [[0, 1], [0, 2], [1, 3], [2, 3]]}
        )
        assert result.status == "ACYCLIC"
        assert result.maximum_edge_count == 2
        assert result.path == (0, 1, 3)

    def test_fork_join_tie_is_independent_of_edge_order(self) -> None:
        """Equal-length suffixes are compared, regardless of insertion order."""
        result = _longest_path(
            {"vertex_count": 4, "edges": [[0, 2], [0, 1], [2, 3], [1, 3]]}
        )
        assert result.maximum_edge_count == 2
        assert result.path == (0, 1, 3)

    def test_disconnected_components(self) -> None:
        """Two disconnected chains: 0->1 and 2->3->4.

        The longest path is in the second component (2 edges).
        """
        result = _longest_path({"vertex_count": 5, "edges": [[0, 1], [2, 3], [3, 4]]})
        assert result.status == "ACYCLIC"
        assert result.maximum_edge_count == 2
        assert result.path == (2, 3, 4)

    def test_cycle_returns_not_applicable(self) -> None:
        """A cyclic graph returns NOT_APPLICABLE."""
        result = _longest_path({"vertex_count": 3, "edges": [[0, 1], [1, 2], [2, 0]]})
        assert result.status == "NOT_APPLICABLE"
        assert result.maximum_edge_count == 0
        assert result.path == ()

    def test_two_node_cycle(self) -> None:
        """A two-node cycle is also NOT_APPLICABLE."""
        result = _longest_path({"vertex_count": 2, "edges": [[0, 1], [1, 0]]})
        assert result.status == "NOT_APPLICABLE"
        assert result.path == ()

    def test_dag_with_back_edge_to_earlier_component(self) -> None:
        """A more complex DAG with multiple path lengths.

        Graph: 0 -> 1 -> 2, 0 -> 3, 3 -> 4 -> 5.
        Longest path is 0 -> 3 -> 4 -> 5 (3 edges).
        But 0 -> 1 -> 2 is only 2 edges.

        Actually: 0 -> 3 -> 4 -> 5 = 3 edges, 0 -> 1 -> 2 = 2 edges.
        """
        result = _longest_path(
            {
                "vertex_count": 6,
                "edges": [[0, 1], [1, 2], [0, 3], [3, 4], [4, 5]],
            }
        )
        assert result.status == "ACYCLIC"
        assert result.maximum_edge_count == 3
        assert result.path == (0, 3, 4, 5)

    def test_source_field_preserves_input_graph(self) -> None:
        """The result source field should carry the original graph."""
        graph = {"vertex_count": 3, "edges": [[0, 1], [1, 2]]}
        result = _longest_path(graph)
        assert result.source.vertex_count == 3
        assert len(result.source.edges) == 2

    def test_single_vertex_path_witness(self) -> None:
        """A graph with no edges still reports a single-vertex path."""
        result = _longest_path({"vertex_count": 2, "edges": []})
        assert result.status == "ACYCLIC"
        assert result.maximum_edge_count == 0
        assert len(result.path) == 1
        assert result.path == (0,)

    def test_convention_label(self) -> None:
        """Verify the convention label is set correctly."""
        result = _longest_path({"vertex_count": 2, "edges": []})
        assert result.convention == "JACOBIAN_DAG_LONGEST_PATH"

    def test_lexicographic_tie_breaking_with_longer_path(self) -> None:
        """When two paths have the same length, the lexicographically least
        vertex sequence is chosen.

        Graph: 0 -> 2 -> 3 (length 2), 1 -> 4 -> 5 (length 2).
        Lexicographically least is (0, 2, 3).
        """
        result = _longest_path(
            {"vertex_count": 6, "edges": [[0, 2], [2, 3], [1, 4], [4, 5]]}
        )
        assert result.status == "ACYCLIC"
        assert result.maximum_edge_count == 2
        assert result.path == (0, 2, 3)
