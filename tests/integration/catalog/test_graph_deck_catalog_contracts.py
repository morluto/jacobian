"""Anonymous graph-deck publication and forged-request contracts."""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation


def test_forged_catalog_request_missing_card_order_is_a_domain_error() -> None:
    tool = Catalog.open().operation("graph.deck.from_cards.construct")
    assert tool is not None
    # A wire payload missing card_order is rejected by request parsing, so the
    # domain guard is only reachable through a caller that bypasses model
    # validation. This forged request deliberately enters at the kernel.
    forged = tool.request_type.model_construct(cards=())
    with pytest.raises(OperationDomainValidationError) as exc_info:
        tool.run(forged)
    assert exc_info.value.errors()[0]["type"] == "graph_deck.anonymous_order_invalid"


def test_catalog_publishes_anonymous_cards_as_distinct_from_realizable_decks() -> None:
    catalog = Catalog.open()
    tool = catalog.operation("graph.deck.from_cards.construct")
    assert tool is not None
    # Route through dispatch so operation-ID lookup, strict wire parsing, and
    # the serialized output envelope all have to accept this request.
    result = invoke_operation(
        tool.operation_id,
        {"card_order": 1, "cards": [{"vertices": ["x"], "edges": []}]},
        catalog,
    )
    validated = tool.result_type.model_validate(result.output)
    assert validated.card_order == 1
    assert validated.classes[0].multiplicity == 1
