"""Catalog boundary for bounded exact Steiner triple-system construction (#1666)."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation

_OPERATION_ID = "combinatorics.design.steiner_triple_system.construct"


def test_discovery_description_states_sts_pair_coverage() -> None:
    tool = Catalog.open().operation(_OPERATION_ID)
    assert tool is not None
    description = tool.description.lower()
    assert "every unordered pair" in description
    assert "exactly one block" in description
    assert "unknown" in description
    assert "exact cover" not in description
    assert "replay" not in description
    assert "search" not in description


def test_constructor_executes_through_public_catalog_boundary() -> None:
    result = invoke_operation(
        _OPERATION_ID,
        {"order": 7, "search_budget": 100_000},
        Catalog.open(),
    )
    assert result.output["outcome"]["status"] == "COMPUTED"
    assert len(result.output["outcome"]["design"]["blocks"]) == 7


def test_non_array_fixed_triples_are_request_validation_errors() -> None:
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(
            _OPERATION_ID,
            {
                "order": 7,
                "search_budget": 100,
                "shard": {"order": 7, "fixed_triples": {}},
            },
            Catalog.open(),
        )
