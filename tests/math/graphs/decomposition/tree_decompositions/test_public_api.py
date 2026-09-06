"""Exact public API contract for jacobian.math.graphs.decomposition.tree_decompositions."""

from __future__ import annotations

from jacobian.math.graphs.decomposition import tree_decompositions


def test_exact_public_api_symbols() -> None:
    expected = (
        "TreeDecomposition",
        "adhesions",
        "bag_intersection_graph",
        "reroot",
        "restrict",
        "verify_adhesions",
        "verify_bag_intersection_graph",
        "verify_reroot",
        "verify_vertex_occurrences",
        "verify_width",
        "vertex_occurrences",
        "width",
    )
    assert tuple(tree_decompositions.__all__) == expected
    assert len(tree_decompositions.__all__) == len(set(tree_decompositions.__all__))
    assert all(not name.startswith("_") for name in tree_decompositions.__all__)
    assert all(
        hasattr(tree_decompositions, name) for name in tree_decompositions.__all__
    )
