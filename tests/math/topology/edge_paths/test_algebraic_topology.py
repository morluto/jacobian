"""Tests for algebraic topology operations."""

import pytest

from jacobian.math.topology.edge_paths._models import (
    EdgePathConcatenateRequest,
    EdgePathWordRequest,
    OrientedEdge,
)
from jacobian.math.topology.edge_paths._tools import TOOLS
from jacobian.math.topology.edge_paths.operations import (
    concatenate_edge_paths,
    edge_path_word,
)


def _word(request: EdgePathWordRequest):
    return edge_path_word(
        request.vertex_count, request.edges, request.start_vertex, request.path
    )


def _concatenate(request: EdgePathConcatenateRequest):
    return concatenate_edge_paths(request.vertex_count, request.path_a, request.path_b)


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
        path=(
            OrientedEdge(edge_index=0, orientation=1),
            OrientedEdge(edge_index=1, orientation=1),
        ),
    )
    result = _word(request)
    assert result.word == ("e1", "e2")
    assert result.length == 2


def test_edge_path_word_backward() -> None:
    request = EdgePathWordRequest(
        vertex_count=3,
        edges=((0, 1), (1, 2), (2, 0)),
        start_vertex=1,
        path=(OrientedEdge(edge_index=0, orientation=-1),),
    )
    result = _word(request)
    assert result.word == ("e1^-1",)


def test_edge_path_concatenate() -> None:
    request = EdgePathConcatenateRequest(
        vertex_count=3,
        path_a=(0, 1),
        path_b=(1, 2),
    )
    result = _concatenate(request)
    assert result.path == (0, 1, 2)
    assert result.length == 3


def test_edge_path_word_rejects_non_edge_path() -> None:
    """Path with a step that is not an edge is rejected by the operation."""
    with pytest.raises(ValueError):
        _word(
            EdgePathWordRequest.model_validate(
                {
                    "vertex_count": 4,
                    "edges": ((0, 1), (2, 3)),
                    "start_vertex": 0,
                    "path": ({"edge_index": 1, "orientation": 1},),
                }
            )
        )


def test_edge_path_word_no_invalid_markers() -> None:
    """The resulting word must only contain valid generator entries."""
    request = EdgePathWordRequest(
        vertex_count=3,
        edges=((0, 1), (1, 2), (2, 0)),
        start_vertex=0,
        path=tuple(OrientedEdge(edge_index=index, orientation=1) for index in range(3)),
    )
    result = _word(request)
    for entry in result.word:
        assert "INVALID" not in entry


def test_parallel_edges_and_loops_have_explicit_identity_and_orientation() -> None:
    parallel = _word(
        EdgePathWordRequest(
            vertex_count=2,
            edges=((0, 1), (0, 1)),
            start_vertex=0,
            path=(OrientedEdge(edge_index=1, orientation=1),),
        )
    )
    assert parallel.word == ("e2",)
    loop = _word(
        EdgePathWordRequest(
            vertex_count=2,
            edges=((0, 0),),
            start_vertex=0,
            path=(OrientedEdge(edge_index=0, orientation=-1),),
        )
    )
    assert loop.word == ("e1^-1",)


def test_edge_path_concatenate_rejects_discontinuous() -> None:
    """Concatenation must require matching endpoints."""
    with pytest.raises(ValueError):
        _concatenate(
            EdgePathConcatenateRequest(
                vertex_count=3,
                path_a=(0, 1),
                path_b=(2, 0),
            )
        )
