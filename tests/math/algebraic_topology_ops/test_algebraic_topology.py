"""Tests for algebraic topology operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.algebraic_topology_ops._models import (
    EdgePathConcatenateRequest,
    EdgePathWordRequest,
)
from jacobian.math.algebraic_topology_ops._operations import (
    compute_edge_path_concatenate,
    compute_edge_path_word,
)
from jacobian.math.algebraic_topology_ops._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "topology.simplicial.edge_path.word.compute",
        "topology.simplicial.edge_path.concatenate.compute",
    }


def test_edge_path_word_forward() -> None:
    request = EdgePathWordRequest(
        vertex_count=3,
        edges=((0, 1), (1, 2), (2, 0)),
        start_vertex=0,
        path=({"edge_index": 0, "orientation": 1}, {"edge_index": 1, "orientation": 1}),
    )
    result = compute_edge_path_word(request)
    assert result.word == ("e1", "e2")
    assert result.length == 2


def test_edge_path_word_backward() -> None:
    request = EdgePathWordRequest(
        vertex_count=3,
        edges=((0, 1), (1, 2), (2, 0)),
        start_vertex=1,
        path=({"edge_index": 0, "orientation": -1},),
    )
    result = compute_edge_path_word(request)
    assert result.word == ("e1^-1",)


def test_edge_path_concatenate() -> None:
    request = EdgePathConcatenateRequest(
        vertex_count=3,
        path_a=(0, 1),
        path_b=(1, 2),
    )
    result = compute_edge_path_concatenate(request)
    assert result.path == (0, 1, 2)
    assert result.length == 3


def test_edge_path_word_rejects_non_edge_path() -> None:
    """Path with a step that is not an edge must be rejected at request level."""
    with pytest.raises(ValidationError):
        EdgePathWordRequest(
            vertex_count=4,
            edges=((0, 1), (2, 3)),
            start_vertex=0,
            path=({"edge_index": 1, "orientation": 1},),
        )


def test_edge_path_word_no_invalid_markers() -> None:
    """The resulting word must only contain valid generator entries."""
    request = EdgePathWordRequest(
        vertex_count=3,
        edges=((0, 1), (1, 2), (2, 0)),
        start_vertex=0,
        path=tuple({"edge_index": index, "orientation": 1} for index in range(3)),
    )
    result = compute_edge_path_word(request)
    for entry in result.word:
        assert "INVALID" not in entry


def test_parallel_edges_and_loops_have_explicit_identity_and_orientation() -> None:
    parallel = compute_edge_path_word(
        EdgePathWordRequest(
            vertex_count=2,
            edges=((0, 1), (0, 1)),
            start_vertex=0,
            path=({"edge_index": 1, "orientation": 1},),
        )
    )
    assert parallel.word == ("e2",)
    loop = compute_edge_path_word(
        EdgePathWordRequest(
            vertex_count=2,
            edges=((0, 0),),
            start_vertex=0,
            path=({"edge_index": 0, "orientation": -1},),
        )
    )
    assert loop.word == ("e1^-1",)


def test_edge_path_concatenate_rejects_discontinuous() -> None:
    """Concatenation must require matching endpoints."""
    with pytest.raises(ValidationError):
        EdgePathConcatenateRequest(
            vertex_count=3,
            path_a=(0, 1),
            path_b=(2, 0),
        )
