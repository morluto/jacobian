"""Catalog discovery and invocation of exact deletion-deck operations."""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationMatchRequest
from jacobian.dispatch import invoke_operation

_EDGE_COUNT = "graph.deck.vertex.edge_count.compute"
_DEGREE_MULTISET = "graph.deck.vertex.degree_multiset.compute"


def test_catalog_finds_vertex_deck_edge_count_operation() -> None:
    found = Catalog.open().match(
        OperationMatchRequest(
            need="reconstruct exact source graph edge count from complete vertex deck",
            limit=5,
        )
    )
    assert found.matches[0].operation_id == _EDGE_COUNT


def test_catalog_finds_vertex_deck_degree_multiset_operation() -> None:
    catalog = Catalog.open()
    found = catalog.match(
        OperationMatchRequest(
            need="reconstruct the exact degree multiset from a complete vertex deck",
            limit=5,
        )
    )
    assert found.matches[0].operation_id == _DEGREE_MULTISET

    operation = catalog.operation(_DEGREE_MULTISET)
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert operation.result_type.model_validate_json(
        json.dumps(result.output)
    ).degrees == (
        0,
        0,
        0,
    )
