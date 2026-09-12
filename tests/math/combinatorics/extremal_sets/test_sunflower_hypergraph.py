"""Complete source-bound sunflower triple construction."""

from jacobian.math.combinatorics.extremal_sets._sunflower_r import (
    construct_sunflower_family,
)
from jacobian.math.combinatorics.extremal_sets.operations import (
    construct_sunflower_hypergraph,
)
from jacobian.math.combinatorics.extremal_sets.values import IndexedFiniteSetFamily


def test_all_sunflower_triples_and_cores_are_retained() -> None:
    source = IndexedFiniteSetFamily(
        ground_set_size=5,
        members=((0, 1), (0, 2), (0, 3), (1, 2)),
    )
    result = construct_sunflower_hypergraph(source)
    assert [(row.source_indices, row.core) for row in result.sunflowers] == [
        ((0, 1, 2), (0,)),
    ]
    assert result.hypergraph.vertices == ("0", "1", "2", "3")
    assert result.hypergraph.edges == (("sunflower_1", ("0", "1", "2")),)


def test_triple_projection_uses_the_family_ordinal_edge_ids() -> None:
    source = IndexedFiniteSetFamily(
        ground_set_size=6,
        members=((0, 1), (0, 2), (0, 4), (0, 5), (1, 2), (4, 5)),
    )
    triples = construct_sunflower_hypergraph(source)
    family = construct_sunflower_family(source, 3)
    assert triples.hypergraph == family.hypergraph
    assert [row.edge_id for row in triples.sunflowers] == [
        row.edge_id for row in family.sunflowers
    ]
    assert [row.edge_id for row in triples.sunflowers] == [
        "sunflower_1",
        "sunflower_2",
        "sunflower_3",
        "sunflower_4",
    ]


def test_empty_family_returns_empty_complete_hypergraph() -> None:
    source = IndexedFiniteSetFamily(ground_set_size=0, members=())
    result = construct_sunflower_hypergraph(source)
    assert result.sunflowers == ()
    assert result.hypergraph.vertices == ()
    assert result.hypergraph.edges == ()
