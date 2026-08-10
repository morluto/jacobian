from __future__ import annotations

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityObligationStatus,
    CapabilityRelationshipStatus,
    CapabilityRequest,
)
from jacobian.contracts.checkers import CheckerDecision
from jacobian.contracts.results import Arithmetic, Conclusion, Coverage, Method


def _request(
    mode: CapabilityMode,
    *,
    missing_last: bool = False,
    require_disjoint: bool = True,
) -> CapabilityRequest:
    return CapabilityRequest(
        capability_id="case.partition.finite",
        mode=mode,
        input={
            "universe": ["0", "1", "2", "3", "4", "5"],
            "cases": [
                {"case_id": "even", "members": ["0", "2", "4"]},
                {
                    "case_id": "odd",
                    "members": ["1", "3"] if missing_last else ["1", "3", "5"],
                },
            ],
            "require_disjoint": require_disjoint,
        },
    )


def test_finite_partition_explore_keeps_coverage_obligation_open(
    attached_complete_runtime,
) -> None:

    result = attached_complete_runtime.core.capabilities.invoke(
        _request(CapabilityMode.EXPLORE)
    )

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["missing"] == []
    assert result.output["verification_record_uri"] is None
    assert result.relationships[0].status is CapabilityRelationshipStatus.PROPOSED
    assert result.obligations[0].status is CapabilityObligationStatus.OPEN


def test_finite_partition_verify_replays_and_discharges_obligation(
    authorized_complete_runtime,
) -> None:

    runtime = authorized_complete_runtime
    result = runtime.core.capabilities.invoke(_request(CapabilityMode.VERIFY))

    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.assurance.verification_record_uri is not None
    assert result.completeness.verification_record_uri == (
        result.assurance.verification_record_uri
    )
    assert result.relationships[0].status is CapabilityRelationshipStatus.VERIFIED
    assert result.obligations[0].status is CapabilityObligationStatus.DISCHARGED


def test_finite_partition_contract_and_result_preserve_semantic_boundary(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    descriptor = next(
        item
        for item in runtime.core.capabilities.catalog().capabilities
        if item.capability_id == "case.partition.finite"
    )
    result = runtime.core.capabilities.invoke(_request(CapabilityMode.VERIFY))

    assert "opaque caller-supplied strings" in descriptor.description
    assert "does not establish their mathematical meaning" in descriptor.description
    assert "external-domain completeness" in result.scope.description
    assert "member/case semantics were not checked" in result.assurance.basis
    assert "member/case semantics" in result.completeness.basis


def test_finite_partition_reports_conditional_disjointness_scope(
    authorized_complete_runtime,
) -> None:
    runtime = authorized_complete_runtime
    request = _request(CapabilityMode.VERIFY, require_disjoint=False)
    request.input["cases"][1]["members"].append("0")

    result = runtime.core.capabilities.invoke(request)

    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.output["overlaps"] == ["0"]
    assert "disjointness was not required" in result.assurance.basis
    assert "disjointness was not required" in result.completeness.basis
    certificate = runtime.core.store.get(result.output["certificate_uri"])
    assert certificate.payload["payload"]["replay"] == (
        "equality-based finite coverage and conditional disjointness"
    )


def test_finite_partition_verify_fails_closed_on_incomplete_cases(
    authorized_complete_runtime,
) -> None:

    runtime = authorized_complete_runtime
    result = runtime.core.capabilities.invoke(
        _request(CapabilityMode.VERIFY, missing_last=True)
    )

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["missing"] == ["5"]
    assert result.output["verification_record_uri"] is None
    assert result.obligations[0].status is CapabilityObligationStatus.OPEN


def test_verification_rejects_checker_obligation_outside_request(
    authorized_complete_runtime,
    monkeypatch,
) -> None:

    runtime = authorized_complete_runtime

    def accept_with_unbound_obligation(
        *,
        request: dict[str, object],
        **_: object,
    ) -> CheckerDecision:
        scope = request["scope"]
        candidate = request["candidate"]
        assert isinstance(scope, dict)
        assert isinstance(candidate, dict)
        return CheckerDecision(
            accepted=True,
            conclusion=Conclusion.TRUE,
            arithmetic=Arithmetic.EXACT_INTEGER,
            method=Method.EXHAUSTIVE_FINITE,
            coverage=Coverage.EXHAUSTIVE,
            relation_id="case.relation.partitions",
            relationship_source_artifact_uris=(str(scope["artifact_uri"]),),
            relationship_target_artifact_uris=(str(candidate["artifact_uri"]),),
            obligation_uri="artifact://sha256/" + "9" * 64,
        )

    monkeypatch.setattr(
        runtime.services.verification,
        "_run_checker",
        accept_with_unbound_obligation,
    )

    result = runtime.core.capabilities.invoke(_request(CapabilityMode.VERIFY))

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["verification_record_uri"] is None
    assert result.obligations[0].status is CapabilityObligationStatus.OPEN


def test_finite_partition_duplicate_case_ids_cannot_report_complete(
    attached_complete_runtime,
) -> None:
    request = _request(CapabilityMode.EXPLORE)
    request.input["cases"][1]["case_id"] = "even"

    result = attached_complete_runtime.core.capabilities.invoke(request)

    assert result.output["duplicate_case_ids"] == ["even"]
    assert result.completeness.status.value == "PARTIAL"
