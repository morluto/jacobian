"""Tests for edge-deletion diameter profile."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.edge_deletion_diameter_profile.operations import (
    edge_deletion_diameter_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices: list[str], edges: list[list[str]]) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices), edges=tuple(tuple(e) for e in edges)
    )


def test_path_three_vertices_disconnected():
    g = _graph(["0", "1", "2"], [["0", "1"], ["1", "2"]])
    result = edge_deletion_diameter_profile(g)
    assert result.source_diameter == 2
    assert len(result.entries) == 2
    for entry in result.entries:
        assert entry.result == "DISCONNECTED"
        assert entry.diameter is None


def test_triangle_becomes_path():
    g = _graph(["0", "1", "2"], [["0", "1"], ["0", "2"], ["1", "2"]])
    result = edge_deletion_diameter_profile(g)
    assert result.source_diameter == 1
    for entry in result.entries:
        assert entry.result == "DIAMETER"
        assert entry.diameter == 2


def test_square_cycle():
    g = _graph(["0", "1", "2", "3"], [["0", "1"], ["0", "3"], ["1", "2"], ["2", "3"]])
    result = edge_deletion_diameter_profile(g)
    assert result.source_diameter == 2
    for entry in result.entries:
        assert entry.result == "DIAMETER"
        # Removing one edge from C4 gives P4 with diameter 3
        assert entry.diameter == 3


def test_disconnected_rejected():
    g = _graph(["0", "1", "2"], [["0", "1"]])
    with pytest.raises(OperationDomainValidationError, match="connected"):
        edge_deletion_diameter_profile(g)


def test_empty_graph_rejected():
    g = SimpleUndirectedGraph(vertices=(), edges=())
    with pytest.raises(OperationDomainValidationError, match=r"nonempty|connected"):
        edge_deletion_diameter_profile(g)


def test_json_round_trip():
    g = _graph(["0", "1", "2"], [["0", "1"], ["1", "2"], ["0", "2"]])
    result = edge_deletion_diameter_profile(g)
    json_val = result.model_dump_json()
    replay = type(result).model_validate_json(json_val, strict=True)
    assert replay == result


def test_complete_graph_remains_connected():
    # K4 diameter 1, after removing one edge diameter remains 1 (still complete minus one edge has diameter 2? Actually K4 minus one edge: two vertices not directly connected but via two hops, so diameter 2)
    g = _graph(
        ["0", "1", "2", "3"],
        [["0", "1"], ["0", "2"], ["0", "3"], ["1", "2"], ["1", "3"], ["2", "3"]],
    )
    result = edge_deletion_diameter_profile(g)
    assert result.source_diameter == 1
    for entry in result.entries:
        assert entry.result == "DIAMETER"
        assert entry.diameter == 2
