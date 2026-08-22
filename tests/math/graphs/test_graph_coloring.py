"""Tests for exact maximal-independent-set decision."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.coloring._models import (
    MaximalIndependentSetRequest,
    MaximalIndependentSetResult,
)
from jacobian.math.graphs.coloring._operations import (
    compute_maximal_independent_set_decision,
)


def _request(
    *,
    vertex_count: int,
    edges: list[list[int]],
    candidate_set: list[int],
) -> MaximalIndependentSetRequest:
    return MaximalIndependentSetRequest.model_validate(
        {
            "graph": {"vertex_count": vertex_count, "edges": edges},
            "candidate_set": candidate_set,
        }
    )


def test_path_candidate_is_maximal() -> None:
    result = compute_maximal_independent_set_decision(
        _request(
            vertex_count=4,
            edges=[[0, 1], [1, 2], [2, 3]],
            candidate_set=[0, 2],
        )
    )

    assert result == MaximalIndependentSetResult(decision="MAXIMAL")


def test_non_independent_candidate_returns_canonical_blocking_edge() -> None:
    result = compute_maximal_independent_set_decision(
        _request(
            vertex_count=3,
            edges=[[1, 0], [1, 2]],
            candidate_set=[0, 1],
        )
    )

    assert result.decision == "NOT_INDEPENDENT"
    assert result.blocking_edge == (0, 1)
    assert result.addable_vertex is None


def test_nonmaximal_candidate_returns_smallest_addable_vertex() -> None:
    result = compute_maximal_independent_set_decision(
        _request(
            vertex_count=4,
            edges=[[0, 1], [1, 2], [2, 3]],
            candidate_set=[0],
        )
    )

    assert result.decision == "INDEPENDENT_NOT_MAXIMAL"
    assert result.blocking_edge is None
    assert result.addable_vertex == 2


def test_empty_candidate_in_nonempty_graph_is_not_maximal() -> None:
    result = compute_maximal_independent_set_decision(
        _request(vertex_count=1, edges=[], candidate_set=[])
    )

    assert result.decision == "INDEPENDENT_NOT_MAXIMAL"
    assert result.addable_vertex == 0


def test_singleton_in_complete_graph_is_maximal() -> None:
    result = compute_maximal_independent_set_decision(
        _request(
            vertex_count=3,
            edges=[[0, 1], [0, 2], [1, 2]],
            candidate_set=[1],
        )
    )

    assert result.decision == "MAXIMAL"


def test_all_vertices_of_empty_graph_form_a_maximal_set() -> None:
    result = compute_maximal_independent_set_decision(
        _request(vertex_count=3, edges=[], candidate_set=[0, 1, 2])
    )

    assert result.decision == "MAXIMAL"


@pytest.mark.parametrize(
    ("candidate_set", "message"),
    [
        ([1, 0], "strictly increasing"),
        ([0, 0], "duplicate"),
        ([0, 3], "0..vertex_count-1"),
    ],
)
def test_candidate_set_is_canonical_and_in_range(
    candidate_set: list[int],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _request(vertex_count=3, edges=[], candidate_set=candidate_set)


def test_result_rejects_witness_for_maximal_decision() -> None:
    with pytest.raises(ValidationError, match="must not carry"):
        MaximalIndependentSetResult(
            decision="MAXIMAL",
            addable_vertex=0,
        )


def test_result_requires_matching_rejection_witness() -> None:
    with pytest.raises(ValidationError, match="blocking edge"):
        MaximalIndependentSetResult(decision="NOT_INDEPENDENT")
    with pytest.raises(ValidationError, match="addable vertex"):
        MaximalIndependentSetResult(decision="INDEPENDENT_NOT_MAXIMAL")


class TestEdgeKColorability:
    def _petersen(self):
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        return SimpleUndirectedGraph(
            vertices=("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
            edges=(
                ("0", "1"),
                ("1", "2"),
                ("2", "3"),
                ("3", "4"),
                ("0", "4"),
                ("5", "7"),
                ("7", "9"),
                ("6", "9"),
                ("6", "8"),
                ("5", "8"),
                ("0", "5"),
                ("1", "6"),
                ("2", "7"),
                ("3", "8"),
                ("4", "9"),
            ),
        )

    def test_petersen_not_3_edge_colorable(self):
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )

        result = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=self._petersen(), colors=3),
        )
        assert result.colorable is False
        assert result.coloring is None

    def test_petersen_4_edge_colorable_and_proper(self):
        from jacobian.math.graphs.coloring._models import (
            EdgeColoringCheckRequest,
            EdgeKColorabilityRequest,
        )
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_coloring_check,
            compute_edge_k_colorability,
        )

        g = self._petersen()
        result = compute_edge_k_colorability(
            EdgeKColorabilityRequest(graph=g, colors=4)
        )
        assert result.colorable is True
        assert result.coloring is not None
        assert len(result.coloring) == 15
        check = compute_edge_coloring_check(
            EdgeColoringCheckRequest(graph=g, colors=4, coloring=result.coloring),
        )
        assert check.proper is True

    def test_triangle_needs_3_edge_colors(self):
        from jacobian.math.graphs.coloring._models import EdgeKColorabilityRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_k_colorability,
        )
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        g = SimpleUndirectedGraph(
            vertices=("0", "1", "2"),
            edges=(("0", "1"), ("1", "2"), ("0", "2")),
        )
        assert (
            compute_edge_k_colorability(
                EdgeKColorabilityRequest(graph=g, colors=2)
            ).colorable
            is False
        )
        assert (
            compute_edge_k_colorability(
                EdgeKColorabilityRequest(graph=g, colors=3)
            ).colorable
            is True
        )


class TestEdgeColoringCheck:
    def _petersen(self):
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        return SimpleUndirectedGraph(
            vertices=("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
            edges=(
                ("0", "1"),
                ("1", "2"),
                ("2", "3"),
                ("3", "4"),
                ("0", "4"),
                ("5", "7"),
                ("7", "9"),
                ("6", "9"),
                ("6", "8"),
                ("5", "8"),
                ("0", "5"),
                ("1", "6"),
                ("2", "7"),
                ("3", "8"),
                ("4", "9"),
            ),
        )

    def test_proper_coloring_accepted(self):
        from jacobian.math.graphs.coloring._models import EdgeColoringCheckRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_coloring_check,
        )

        g = self._petersen()
        coloring = (1, 0, 1, 3, 2, 3, 0, 3, 1, 2, 0, 2, 2, 0, 1)
        result = compute_edge_coloring_check(
            EdgeColoringCheckRequest(graph=g, colors=4, coloring=coloring),
        )
        assert result.proper is True
        assert result.blocking_edge is None

    def test_improper_coloring_reports_blocking_edge(self):
        from jacobian.math.graphs.coloring._models import EdgeColoringCheckRequest
        from jacobian.math.graphs.coloring._operations import (
            compute_edge_coloring_check,
        )
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        g = SimpleUndirectedGraph(
            vertices=("0", "1", "2"),
            edges=(("0", "1"), ("1", "2")),
        )
        # edges (0,1) and (1,2) share vertex 1; same color is improper.
        result = compute_edge_coloring_check(
            EdgeColoringCheckRequest(graph=g, colors=2, coloring=(0, 0)),
        )
        assert result.proper is False
        assert result.blocking_edge is not None
        assert result.conflicting_edge is not None

    def test_rejects_wrong_length_assignment(self):
        from jacobian.math.graphs.coloring._models import EdgeColoringCheckRequest
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        g = SimpleUndirectedGraph(
            vertices=("0", "1"),
            edges=(("0", "1"),),
        )
        with pytest.raises(ValidationError, match="one color per edge"):
            EdgeColoringCheckRequest(graph=g, colors=2, coloring=(0, 1))
