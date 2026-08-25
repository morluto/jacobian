from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationBrowseCard,
    OperationBrowseResult,
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationDiscoveryRequest,
    OperationDiscoveryResult,
)


def _descriptor(operation_id: str) -> dict[str, object]:
    return {
        "operation_id": operation_id,
        "title": operation_id,
        "description": "A bounded test operation.",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    }


def _error_type(error: ValidationError) -> str:
    return error.errors()[0]["type"]


def test_catalog_discovery_text_has_no_arbitrary_character_cap() -> None:
    long_text = "x" * 513

    request = OperationDiscoveryRequest(query=long_text)
    descriptor = OperationDescriptor(
        **{**_descriptor("integer.compute.gcd"), "description": long_text}
    )
    match = OperationDiscoveryMatch(
        operation_id="integer.compute.gcd",
        title="Compute gcd",
        description=long_text,
        relevance_score=0,
        applicability="NEEDS_MORE_TYPED_REQUIREMENTS",
        applicability_code="FULL_REQUEST_REQUIRED",
    )
    card = OperationBrowseCard(
        operation_id="integer.compute.gcd",
        title="Compute gcd",
        description=long_text,
    )

    assert request.query == long_text
    assert descriptor.description == long_text
    assert match.description == long_text
    assert card.description == long_text


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
    with pytest.raises(ValidationError) as error:
        OperationDiscoveryResult.model_validate(
            {**base, "truncated": True, "next_cursor": None}
        )
    assert _error_type(error.value) == "catalog.cursor_state"
    with pytest.raises(ValidationError) as error:
        OperationDiscoveryResult.model_validate(
            {
                **base,
                "truncated": True,
                "next_cursor": "integer.compute.lcm",
            }
        )
    assert _error_type(error.value) == "catalog.cursor_position"


def test_browse_page_metadata_requires_sorted_compact_operation_cards() -> None:
    base = {
        "operations": [
            {
                "operation_id": "integer.compute.gcd",
                "title": "Compute gcd",
                "description": "Compute one exact gcd.",
            }
        ],
        "total_operations": 2,
    }
    with pytest.raises(ValidationError) as error:
        OperationBrowseResult.model_validate(
            {**base, "truncated": True, "next_cursor": None}
        )
    assert _error_type(error.value) == "catalog.cursor_state"
    with pytest.raises(ValidationError) as error:
        OperationBrowseResult.model_validate(
            {
                **base,
                "operations": [
                    *base["operations"],
                    {
                        "operation_id": "integer.compute.factorial",
                        "title": "Compute factorial",
                        "description": "Compute one exact factorial.",
                    },
                ],
                "total_operations": 2,
                "truncated": False,
            }
        )
    assert _error_type(error.value) == "catalog.browse_order"


def test_catalog_rejects_duplicate_or_nondeterministic_operation_ids() -> None:
    with pytest.raises(ValidationError) as error:
        OperationCatalogSnapshot.model_validate(
            {
                "operations": [
                    _descriptor("integer.compute.lcm"),
                    _descriptor("integer.compute.gcd"),
                ],
            }
        )
    assert _error_type(error.value) == "catalog.operation_order"
