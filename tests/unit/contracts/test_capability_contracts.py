from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDiscoveryBrowseRecoveryPath,
    CapabilityDiscoveryInspectCatalogRecoveryPath,
    CapabilityDiscoveryRecoveryPath,
    CapabilityDiscoveryReformulateQueryRecoveryPath,
    CapabilityDiscoveryRemoveFiltersRecoveryPath,
    CapabilityDiscoveryRemoveUnknownDomainRecoveryPath,
    CapabilityMode,
    CapabilityObligation,
    CapabilityObligationStatus,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.results import Execution, ExecutionStatus

RECORD_URI = "artifact://sha256/" + "a" * 64


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


def test_explore_lane_cannot_claim_verified_assurance() -> None:
    with pytest.raises(ValidationError, match="exploration lane"):
        CapabilityResult(
            capability_id="example.solve",
            capability_version="1",
            mode=CapabilityMode.EXPLORE,
            execution=Execution(status=ExecutionStatus.COMPLETED),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.VERIFIED,
                basis="untrusted adapter claim",
                verification_record_uri=RECORD_URI,
            ),
        )


def test_nonverified_assurance_cannot_smuggle_a_record_uri() -> None:
    with pytest.raises(ValidationError, match="only verified"):
        CapabilityAssurance(
            level=CapabilityAssuranceLevel.COMPUTED,
            basis="ordinary deterministic computation",
            verification_record_uri=RECORD_URI,
        )


def test_complete_result_requires_an_explicit_scope() -> None:
    with pytest.raises(
        ValidationError, match="complete result requires explicit scope"
    ):
        CapabilityResult(
            capability_id="graph.enumerate.nonisomorphic",
            capability_version="1",
            mode=CapabilityMode.EXPLORE,
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
            mode=CapabilityMode.EXPLORE,
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
            mode=CapabilityMode.VERIFY,
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
            mode=CapabilityMode.EXPLORE,
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
