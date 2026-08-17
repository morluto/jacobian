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
