from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationBrowseCard,
    OperationBrowseResult,
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryMatch,
    OperationMatchRequest,
    OperationMatchResult,
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


def test_catalog_need_accepts_a_concise_mathematical_description() -> None:
    long_text = "x" * 513

    request = OperationMatchRequest(need=long_text)
    descriptor = OperationDescriptor(
        operation_id="integer.compute.gcd",
        title="integer.compute.gcd",
        description=long_text,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    match = OperationDiscoveryMatch(
        operation_id="integer.compute.gcd",
        title="Compute gcd",
        description=long_text,
    )
    card = OperationBrowseCard(
        operation_id="integer.compute.gcd",
        title="Compute gcd",
        description=long_text,
    )

    assert request.need == long_text
    assert descriptor.description == long_text
    assert match.description == long_text
    assert card.description == long_text


def test_discovery_page_metadata_is_bound_to_returned_matches() -> None:
    base = {
        "need": "gcd",
        "matches": [
            {
                "operation_id": "integer.compute.gcd",
                "title": "Compute gcd",
                "description": "Compute one exact gcd.",
            }
        ],
        "total_matches": 2,
    }
    with pytest.raises(ValidationError) as error:
        OperationMatchResult.model_validate(
            {**base, "next_cursor": "integer.compute.lcm"}
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
            {**base, "next_cursor": "integer.compute.lcm"}
        )
    assert _error_type(error.value) == "catalog.cursor_position"
    with pytest.raises(ValidationError) as error:
        OperationBrowseResult.model_validate(
            {
                **base,
                "operations": [
                    {
                        "operation_id": "integer.compute.gcd",
                        "title": "Compute gcd",
                        "description": "Compute one exact gcd.",
                    },
                    {
                        "operation_id": "integer.compute.factorial",
                        "title": "Compute factorial",
                        "description": "Compute one exact factorial.",
                    },
                ],
                "total_operations": 2,
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


def test_descriptor_rejects_unbounded_discovery_vocabulary() -> None:
    with pytest.raises(ValidationError):
        OperationDescriptor.model_validate(
            {
                **_descriptor("integer.compute.gcd"),
                "discovery_terms": [f"term_{index}" for index in range(9)],
            }
        )
