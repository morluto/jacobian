from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCatalog,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDiscoveryBrowseRecoveryPath,
    CapabilityDiscoveryInspectCatalogRecoveryPath,
    CapabilityDiscoveryRecoveryPath,
    CapabilityDiscoveryReformulateQueryRecoveryPath,
    CapabilityDiscoveryRemoveFiltersRecoveryPath,
    CapabilityDiscoveryRemoveUnknownDomainRecoveryPath,
    CapabilityDiscoveryResult,
    CapabilityObligation,
    CapabilityObligationStatus,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityResult,
    CapabilityScope,
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


def test_discovery_recovery_paths_are_closed_discriminated_contracts() -> None:
    paths: tuple[CapabilityDiscoveryRecoveryPath, ...] = (
        CapabilityDiscoveryReformulateQueryRecoveryPath(action="reformulate_query"),
        CapabilityDiscoveryRemoveUnknownDomainRecoveryPath(
            action="remove_unknown_domain_filter", rejected_domain="arithmetic"
        ),
        CapabilityDiscoveryRemoveFiltersRecoveryPath(action="remove_filters"),
        CapabilityDiscoveryBrowseRecoveryPath(action="browse"),
        CapabilityDiscoveryInspectCatalogRecoveryPath(action="inspect_catalog"),
    )
    adapter = TypeAdapter(CapabilityDiscoveryRecoveryPath)
    schema = adapter.json_schema()

    assert {path.action for path in paths} == {
        "reformulate_query",
        "remove_unknown_domain_filter",
        "remove_filters",
        "browse",
        "inspect_catalog",
    }
    assert schema["discriminator"]["propertyName"] == "action"
    assert all(
        "action" in definition["required"] for definition in schema["$defs"].values()
    )
    assert CapabilityDiscoveryBrowseRecoveryPath(action="browse").model_dump(
        mode="json"
    ) == {
        "action": "browse",
        "tool": "math.find",
        "arguments": {},
    }
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "action": "browse",
                "tool": "math.find",
                "arguments": {"limit": 5},
            }
        )
    with pytest.raises(ValidationError):
        adapter.validate_python({"action": "recommended_next_step"})
    tagless_path = {"arguments": {}}
    assert list(Draft202012Validator(schema).iter_errors(tagless_path))
    with pytest.raises(ValidationError):
        adapter.validate_python(tagless_path)


def test_discovery_page_metadata_is_bound_to_returned_matches() -> None:
    base = {
        "routing_basis": "The request uses a compatible structured input.",
        "matches": [
            {
                "capability_id": "integer.compute.gcd",
                "title": "Compute gcd",
                "description": "Compute one exact gcd.",
            }
        ],
        "total_matches": 2,
        "portfolio_fit_basis": "One lexical candidate was found.",
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


def test_nonverified_assurance_cannot_smuggle_a_record_uri() -> None:
    with pytest.raises(ValidationError, match="only verified"):
        CapabilityAssurance(
            level=CapabilityAssuranceLevel.COMPUTED,
            basis="ordinary deterministic computation",
            verification_record_uri=RECORD_URI,
        )


def test_verified_result_publishes_its_record_as_a_first_class_artifact() -> None:
    with pytest.raises(ValidationError, match="included in artifact_uris"):
        CapabilityResult(
            capability_id="example.verify",
            capability_version="1",
            execution=Execution(status=ExecutionStatus.COMPLETED),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.VERIFIED,
                basis="independent checker accepted the claim",
                verification_record_uri=RECORD_URI,
            ),
        )

    result = CapabilityResult(
        capability_id="example.verify",
        capability_version="1",
        execution=Execution(status=ExecutionStatus.COMPLETED),
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.VERIFIED,
            basis="independent checker accepted the claim",
            verification_record_uri=RECORD_URI,
        ),
        artifact_uris=(RECORD_URI,),
    )
    assert result.artifact_uris == (RECORD_URI,)


def test_complete_result_requires_an_explicit_scope() -> None:
    with pytest.raises(
        ValidationError, match="complete result requires explicit scope"
    ):
        CapabilityResult(
            capability_id="graph.enumerate.nonisomorphic",
            capability_version="1",
            execution=Execution(status=ExecutionStatus.COMPLETED),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis="enumerator reported exhaustion",
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="deterministic enumeration",
            ),
        )


def test_failed_execution_cannot_claim_completeness() -> None:
    with pytest.raises(ValidationError, match="failed execution cannot be complete"):
        CapabilityResult(
            capability_id="graph.enumerate.nonisomorphic",
            capability_version="1",
            execution=Execution(status=ExecutionStatus.TIMEOUT),
            scope=CapabilityScope(
                description="simple graphs on five vertices",
                parameters={"vertices": 5},
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.COMPLETE,
                basis="adapter reached its configured limit",
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.HEURISTIC,
                basis="enumeration timed out",
            ),
        )


def test_verified_relationship_must_use_result_checker_record() -> None:
    other_record = "artifact://sha256/" + "b" * 64
    with pytest.raises(
        ValidationError,
        match="verified relationship must use the result verification record",
    ):
        CapabilityResult(
            capability_id="claim.derive.specialization",
            capability_version="1",
            execution=Execution(status=ExecutionStatus.COMPLETED),
            relationships=(
                CapabilityRelationship(
                    relation_id="claim.relation.specialization",
                    source_artifact_uris=("artifact://sha256/" + "c" * 64,),
                    target_artifact_uris=("artifact://sha256/" + "d" * 64,),
                    status=CapabilityRelationshipStatus.VERIFIED,
                    verification_record_uri=other_record,
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.VERIFIED,
                basis="independent checker accepted the relation",
                verification_record_uri=RECORD_URI,
            ),
        )


def test_discharged_obligation_requires_verified_result() -> None:
    with pytest.raises(
        ValidationError,
        match="discharged obligation requires verified result assurance",
    ):
        CapabilityResult(
            capability_id="case.partition.finite",
            capability_version="1",
            execution=Execution(status=ExecutionStatus.COMPLETED),
            obligations=(
                CapabilityObligation(
                    obligation_uri="artifact://sha256/" + "e" * 64,
                    status=CapabilityObligationStatus.DISCHARGED,
                    verification_record_uri=RECORD_URI,
                ),
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis="partition was generated but not independently checked",
            ),
        )
