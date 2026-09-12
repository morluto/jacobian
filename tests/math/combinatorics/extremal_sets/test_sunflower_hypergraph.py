"""Sunflower construction is published only as the petal-count family operation."""

from jacobian.math.combinatorics.extremal_sets import (
    construct_binary_union_relation,
    construct_sunflower_family,
)
from jacobian.math.combinatorics.extremal_sets.operations import __all__ as operations_all
from jacobian.math.combinatorics.extremal_sets.values import IndexedFiniteSetFamily


def test_operations_module_does_not_export_a_native_sunflower_constructor() -> None:
    assert tuple(operations_all) == ("construct_binary_union_relation",)
    assert construct_binary_union_relation.__name__ == "construct_binary_union_relation"


def test_all_sunflower_triples_and_cores_are_retained() -> None:
    source = IndexedFiniteSetFamily(
        ground_set_size=5,
        members=((0, 1), (0, 2), (0, 3), (1, 2)),
    )
    result = construct_sunflower_family(source, 3)
    assert [(row.source_indices, row.core) for row in result.sunflowers] == [
        ((0, 1, 2), (0,)),
    ]
    assert result.hypergraph.vertices == ("0", "1", "2", "3")
    assert result.hypergraph_edges == (("sunflower_0_1_2", ("0", "1", "2")),)


def test_empty_family_returns_empty_complete_hypergraph() -> None:
    source = IndexedFiniteSetFamily(ground_set_size=0, members=())
    result = construct_sunflower_family(source, 3)
    assert result.sunflowers == ()
    assert result.hypergraph.vertices == ()
    assert result.hypergraph_edges == ()
