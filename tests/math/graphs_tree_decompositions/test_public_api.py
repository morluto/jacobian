"""Exact public API contract for jacobian.math.graphs.tree_decompositions."""

from __future__ import annotations

from jacobian.math.graphs import tree_decompositions


def test_exact_public_api_symbols() -> None:
    expected = (
        "TreeDecomposition",
        "adhesions",
        "bag_intersection_graph",
        "reroot",
        "restrict",
        "vertex_occurrences",
        "width",
    )
    assert tuple(tree_decompositions.__all__) == expected
    assert len(tree_decompositions.__all__) == len(set(tree_decompositions.__all__))
    assert all(not name.startswith("_") for name in tree_decompositions.__all__)
    assert all(
        hasattr(tree_decompositions, name) for name in tree_decompositions.__all__
    )
