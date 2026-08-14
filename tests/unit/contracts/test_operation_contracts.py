from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.operations import (
    OperationCatalogSnapshot,
    OperationDescriptor,
    OperationDiscoveryResult,
    OperationResult,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.provider_runtime import known_provider_runtime

RECORD_URI = "artifact://sha256/" + "a" * 64
POLICY_DIGEST = "sha256:" + "b" * 64


def _descriptor(operation_id: str) -> dict[str, object]:
    return {
        "operation_id": operation_id,
        "version": "1",
        "title": operation_id,
        "description": "A bounded test operation.",
        "provider": "test",
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
                "policy_profile": "DEFAULT",
                "policy_digest": POLICY_DIGEST,
                "operations": [
                    _descriptor("integer.compute.lcm"),
                    _descriptor("integer.compute.gcd"),
                ],
            }
        )


def test_operation_descriptor_does_not_publish_execution_identity() -> None:
    descriptor = OperationDescriptor.model_validate(
        {
            **_descriptor("integer.compute.gcd"),
            "provider": "test.runtime",
            "provider_runtime": known_provider_runtime("test.runtime"),
        }
    )

    assert "provider_runtime" not in descriptor.model_dump(mode="json")
    assert (
        "provider_runtime" not in OperationDescriptor.model_json_schema()["properties"]
    )


def test_noncompleted_execution_cannot_carry_a_verification_record() -> None:
    with pytest.raises(ValidationError, match="cannot carry a verification record"):
        OperationResult(
            operation_id="example.verify",
            operation_version="1",
            execution=Execution(status=ExecutionStatus.TIMEOUT),
            verification_record_uri=RECORD_URI,
        )


def test_verified_result_publishes_its_record_as_a_first_class_artifact() -> None:
    with pytest.raises(ValidationError, match="included in artifact_uris"):
        OperationResult(
            operation_id="example.verify",
            operation_version="1",
            execution=Execution(status=ExecutionStatus.COMPLETED),
            verification_record_uri=RECORD_URI,
        )

    result = OperationResult(
        operation_id="example.verify",
        operation_version="1",
        execution=Execution(status=ExecutionStatus.COMPLETED),
        verification_record_uri=RECORD_URI,
        artifact_uris=(RECORD_URI,),
    )
    assert result.artifact_uris == (RECORD_URI,)
