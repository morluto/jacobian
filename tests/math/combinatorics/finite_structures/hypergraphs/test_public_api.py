"""Public native API for finite hypergraphs."""

from jacobian.math.combinatorics.finite_structures import hypergraphs


def test_hypergraph_public_api_is_explicit() -> None:
    expected = (
        "EdgeIntersectionEntry",
        "EdgeIntersectionsResult",
        "FiniteHypergraph",
        "edge_intersections",
    )

    assert tuple(hypergraphs.__all__) == expected
    assert len(hypergraphs.__all__) == len(set(hypergraphs.__all__))
    assert all(not name.startswith("_") for name in hypergraphs.__all__)
    assert all(hasattr(hypergraphs, name) for name in hypergraphs.__all__)
