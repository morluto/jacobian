"""Complete bounded-cardinality minimal transversal enumeration."""

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._transversal_enumeration import (
    MinimalTransversalEnumerationRequest,
    enumerate_minimal_transversals,
)


def test_crossing_edges_have_singleton_and_two_vertex_minima() -> None:
    source = FiniteHypergraph(
        vertices=("a", "b", "c"),
        edges=(("e0", ("a", "b")), ("e1", ("b", "c"))),
    )
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=2)
    )
    assert result.transversals == (("b",), ("a", "c"))


def test_edge_free_hypergraph_has_one_empty_minimal_transversal() -> None:
    source = FiniteHypergraph(vertices=("a", "b"), edges=())
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=0)
    )
    assert result.transversals == ((),)


def test_empty_edge_has_no_transversal() -> None:
    source = FiniteHypergraph(vertices=("a",), edges=(("empty", ()),))
    result = enumerate_minimal_transversals(
        MinimalTransversalEnumerationRequest(hypergraph=source, maximum_cardinality=1)
    )
    assert result.transversals == ()
