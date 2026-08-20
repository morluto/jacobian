"""Tests for chip-firing operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.chip_firing._models import (
    FiringRequest,
    LaplacianRequest,
)
from jacobian.math.chip_firing._operations import compute_firing, compute_laplacian

GRAPH = {"vertices": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]}


class TestLaplacian:
    def test_path_graph(self) -> None:
        result = compute_laplacian(LaplacianRequest(graph=GRAPH))
        assert result.vertices == ("a", "b", "c")
        assert result.degrees == (1, 2, 1)
        assert result.laplacian == ((1, -1, 0), (-1, 2, -1), (0, -1, 1))

    def test_single_vertex(self) -> None:
        result = compute_laplacian(
            LaplacianRequest(graph={"vertices": ["x"], "edges": []})
        )
        assert result.laplacian == ((0,),)

    def test_triangle(self) -> None:
        result = compute_laplacian(
            LaplacianRequest(
                graph={
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                }
            )
        )
        assert result.degrees == (2, 2, 2)
        assert result.laplacian == ((2, -1, -1), (-1, 2, -1), (-1, -1, 2))


class TestFiring:
    def test_fire_middle_vertex(self) -> None:
        result = compute_firing(
            FiringRequest(graph=GRAPH, divisor=[3, 0, 1], firing_vertex="b")
        )
        # b has degree 2, loses 2 chips: 0-2=-2, each neighbor gains 1: a+1, c+1
        assert result.fired_divisor == (4, -2, 2)

    def test_fire_leaf(self) -> None:
        result = compute_firing(
            FiringRequest(graph=GRAPH, divisor=[3, 0, 1], firing_vertex="a")
        )
        # a has degree 1, loses 1 chip: 3-1=2, neighbor b gains 1: 0+1=1
        assert result.fired_divisor == (2, 1, 1)

    def test_invalid_vertex(self) -> None:
        with pytest.raises(ValidationError, match="firing vertex"):
            FiringRequest(graph=GRAPH, divisor=[0, 0, 0], firing_vertex="x")

    def test_wrong_divisor_length(self) -> None:
        with pytest.raises(ValidationError, match="divisor length"):
            FiringRequest(graph=GRAPH, divisor=[0, 0], firing_vertex="a")
