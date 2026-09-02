from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.tree_enumeration.free_tree_enumeration._models import (
    MAX_ORDER,
    FreeTreeEnumerationRequest,
)
from jacobian.math.graphs.tree_enumeration.free_tree_enumeration.operations import (
    enumerate_free_trees,
)


def test_order_0() -> None:
    result = enumerate_free_trees(0)
    assert result.order == 0
    assert result.count == 0
    assert result.trees == ()


def test_order_1() -> None:
    result = enumerate_free_trees(1)
    assert result.count == 1
    assert len(result.trees) == 1
    assert result.trees[0].vertices == ("0",)
    assert result.trees[0].edges == ()


def test_order_2() -> None:
    result = enumerate_free_trees(2)
    assert result.count == 1
    assert len(result.trees) == 1


def test_order_3() -> None:
    result = enumerate_free_trees(3)
    assert result.count == 1


def test_order_4() -> None:
    result = enumerate_free_trees(4)
    assert result.count == 2


def test_order_5() -> None:
    result = enumerate_free_trees(5)
    assert result.count == 3


def test_order_6() -> None:
    result = enumerate_free_trees(6)
    assert result.count == 6


def test_all_trees_are_valid() -> None:
    """Every returned graph must be a tree: connected and |E| = |V| - 1."""
    for order in range(1, 8):
        result = enumerate_free_trees(order)
        for tree in result.trees:
            assert len(tree.vertices) == order
            assert len(tree.edges) == order - 1


def test_no_isomorphic_pairs() -> None:
    """No two returned trees should be isomorphic."""
    import networkx as nx

    for order in range(1, 8):
        result = enumerate_free_trees(order)
        canonical_forms = []
        for tree in result.trees:
            g: nx.Graph[str] = nx.Graph()
            for v in tree.vertices:
                g.add_node(v)
            for u, v in tree.edges:
                g.add_edge(u, v)
            canonical = nx.weisfeiler_lehman_graph_hash(g)
            canonical_forms.append(canonical)
        assert len(set(canonical_forms)) == len(canonical_forms)


def test_result_preserves_order() -> None:
    result = enumerate_free_trees(5)
    assert result.order == 5
    assert result.count == 3


def test_request_rejects_first_undeliverable_order() -> None:
    with pytest.raises(ValidationError):
        FreeTreeEnumerationRequest(order=MAX_ORDER + 1)

    with pytest.raises(OperationDomainValidationError, match="orders from 0 through"):
        enumerate_free_trees(MAX_ORDER + 1)
