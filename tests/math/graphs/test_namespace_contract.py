"""Owner-local exact public API contract for graphs."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.graphs")
    expected = (
        "GraphCompositionInput",
        "IndependenceNumberBudget",
        "IndependenceNumberRequest",
        "IndependenceNumberResult",
        "SimpleUndirectedGraph",
        "biconnected_components",
        "complement",
        "compose_graphs",
        "diameter",
        "explicit_graph",
        "graph_power",
        "independence_number",
        "induced_subgraph",
        "is_eulerian",
        "line_graph",
        "radius",
        "strongly_connected_components",
        "triangle_count",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
