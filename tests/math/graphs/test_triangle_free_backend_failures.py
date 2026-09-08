"""Operational failures do not disprove triangle-freeness."""

import networkx as nx
import pytest

from jacobian.math.graphs.triangle_free_diameter_augmentation.operations import (
    triangle_free_diameter_augmentation,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_triangle_check_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("triangle backend failure")

    monkeypatch.setattr(nx, "triangles", fail)
    with pytest.raises(RuntimeError, match="triangle backend failure"):
        triangle_free_diameter_augmentation(graph, 1)
