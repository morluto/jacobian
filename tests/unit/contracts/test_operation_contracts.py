from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDiscoveryResult,
)


def _descriptor(operation_id: str) -> dict[str, object]:
    return {
        "operation_id": operation_id,
        "version": "1",
        "title": operation_id,
        "description": "A bounded test operation.",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    }


def test_discovery_page_metadata_is_bound_to_returned_matches() -> None:
    base = {
        "query": "gcd",
        "matches": [
            {
                "operation_id": "integer.compute.gcd",
                "title": "Compute gcd",
                "description": "Compute one exact gcd.",
                "relevance_score": 12,
                "applicability": "NEEDS_MORE_TYPED_REQUIREMENTS",
                "applicability_code": "FULL_REQUEST_REQUIRED",
            }
        ],
        "total_matches": 2,
    }
    with pytest.raises(ValidationError, match="truncated must agree"):
        OperationDiscoveryResult.model_validate(
            {**base, "truncated": True, "next_cursor": None}
        )
    with pytest.raises(ValidationError, match="final returned match"):
        OperationDiscoveryResult.model_validate(
            {
                **base,
                "truncated": True,
                "next_cursor": "integer.compute.lcm",
            }
        )


def test_catalog_rejects_duplicate_or_nondeterministic_operation_ids() -> None:
    with pytest.raises(ValidationError, match="unique and sorted"):
        OperationCatalogSnapshot.model_validate(
            {
                "operations": [
                    _descriptor("integer.compute.lcm"),
                    _descriptor("integer.compute.gcd"),
                ],
            }
        )
