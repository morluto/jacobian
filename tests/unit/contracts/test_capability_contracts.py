from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.capabilities import (
    CapabilityCatalog,
    CapabilityDiscoveryResult,
    CapabilityResult,
)
from jacobian.contracts.results import Execution, ExecutionStatus

RECORD_URI = "artifact://sha256/" + "a" * 64
POLICY_DIGEST = "sha256:" + "b" * 64


def _descriptor(capability_id: str) -> dict[str, object]:
    return {
        "capability_id": capability_id,
        "version": "1",
        "title": capability_id,
        "description": "A bounded test capability.",
        "provider": "test",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
    }


def test_discovery_page_metadata_is_bound_to_returned_matches() -> None:
    base = {
        "query": "gcd",
        "matches": [
            {
                "capability_id": "integer.compute.gcd",
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
        CapabilityDiscoveryResult.model_validate(
            {**base, "truncated": True, "next_cursor": None}
        )
    with pytest.raises(ValidationError, match="final returned match"):
        CapabilityDiscoveryResult.model_validate(
            {
                **base,
                "truncated": True,
                "next_cursor": "integer.compute.lcm",
            }
        )


def test_catalog_rejects_duplicate_or_nondeterministic_capability_ids() -> None:
    with pytest.raises(ValidationError, match="unique and sorted"):
        CapabilityCatalog.model_validate(
            {
                "policy_profile": "DEFAULT",
                "policy_digest": POLICY_DIGEST,
                "capabilities": [
                    _descriptor("integer.compute.lcm"),
                    _descriptor("integer.compute.gcd"),
                ],
            }
        )


def test_noncompleted_execution_cannot_carry_a_verification_record() -> None:
    with pytest.raises(ValidationError, match="cannot carry a verification record"):
        CapabilityResult(
            capability_id="example.verify",
            capability_version="1",
            execution=Execution(status=ExecutionStatus.TIMEOUT),
            verification_record_uri=RECORD_URI,
        )


def test_verified_result_publishes_its_record_as_a_first_class_artifact() -> None:
    with pytest.raises(ValidationError, match="included in artifact_uris"):
        CapabilityResult(
            capability_id="example.verify",
            capability_version="1",
            execution=Execution(status=ExecutionStatus.COMPLETED),
            verification_record_uri=RECORD_URI,
        )

    result = CapabilityResult(
        capability_id="example.verify",
        capability_version="1",
        execution=Execution(status=ExecutionStatus.COMPLETED),
        verification_record_uri=RECORD_URI,
        artifact_uris=(RECORD_URI,),
    )
    assert result.artifact_uris == (RECORD_URI,)
