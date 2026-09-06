"""Public native API for finite hypergraphs."""

from jacobian.math.combinatorics.finite_structures import hypergraphs


def test_hypergraph_public_api_is_explicit() -> None:
    expected = (
        "EdgeIntersectionEntry",
        "EdgeIntersectionsResult",
        "FiniteHypergraph",
        "clique_expansion",
        "dual",
        "edge_intersection_graph",
        "edge_intersections",
        "incidence_graph",
        "independence_number",
        "induced_type_profile",
        "maximum_edge_matching",
        "maximum_weight_packing",
        "minimum_transversal",
        "parameters",
        "verify_independence_number",
        "verify_maximum_edge_matching",
        "verify_minimum_transversal",
        "verify_weighted_packing",
        "vertex_degrees",
    )

    assert tuple(hypergraphs.__all__) == expected
    assert len(hypergraphs.__all__) == len(set(hypergraphs.__all__))
    assert all(not name.startswith("_") for name in hypergraphs.__all__)
    assert all(hasattr(hypergraphs, name) for name in hypergraphs.__all__)
