from __future__ import annotations

from tests.support.core_capability_harnesses import FinitePartitionTestServices

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityObligationStatus,
    CapabilityRelationshipStatus,
    CapabilityRequest,
)


def _request(
    *,
    verify: bool = False,
    missing_last: bool = False,
    require_disjoint: bool = True,
) -> CapabilityRequest:
    return CapabilityRequest(
        capability_id=(
            "case.partition.finite.verify" if verify else "case.partition.finite"
        ),
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


def test_finite_partition_produce_keeps_coverage_obligation_open(
    unauthorized_finite_partition_services: FinitePartitionTestServices,
) -> None:
    result = unauthorized_finite_partition_services.services.core.capabilities.invoke(
        _request()
    )

    assert result.capability_id == "case.partition.finite"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["missing"] == []
    assert result.output["verification_record_uri"] is None
    assert result.relationships[0].status is CapabilityRelationshipStatus.PROPOSED
    assert result.obligations[0].status is CapabilityObligationStatus.OPEN


def test_finite_partition_verify_replays_and_discharges_obligation(
    finite_partition_services: FinitePartitionTestServices,
) -> None:
    runtime = finite_partition_services.services
    result = runtime.core.capabilities.invoke(_request(verify=True))

    assert result.capability_id == "case.partition.finite.verify"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.assurance.verification_record_uri is not None
    assert result.completeness.verification_record_uri == (
        result.assurance.verification_record_uri
    )
    assert result.relationships[0].status is CapabilityRelationshipStatus.VERIFIED
    assert result.obligations[0].status is CapabilityObligationStatus.DISCHARGED


def test_finite_partition_contract_and_result_preserve_semantic_boundary(
    finite_partition_services: FinitePartitionTestServices,
) -> None:
    runtime = finite_partition_services.services
    producer = next(
        item
        for item in runtime.core.capabilities.catalog().capabilities
        if item.capability_id == "case.partition.finite"
    )
    result = runtime.core.capabilities.invoke(_request(verify=True))

    assert "opaque caller-supplied strings" in producer.description
    assert "external-domain completeness" in result.scope.description
    assert "member/case semantics were not checked" in result.assurance.basis
    assert "member/case semantics" in result.completeness.basis


def test_finite_partition_reports_conditional_disjointness_scope(
    finite_partition_services: FinitePartitionTestServices,
) -> None:
    runtime = finite_partition_services.services
    request = _request(verify=True, require_disjoint=False)
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
    finite_partition_services: FinitePartitionTestServices,
) -> None:
    runtime = finite_partition_services.services
    result = runtime.core.capabilities.invoke(_request(verify=True, missing_last=True))

    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["missing"] == ["5"]
    assert result.output["verification_record_uri"] is None
    assert result.obligations[0].status is CapabilityObligationStatus.OPEN


def test_finite_partition_duplicate_case_ids_cannot_report_complete(
    unauthorized_finite_partition_services: FinitePartitionTestServices,
) -> None:
    request = _request()
    request.input["cases"][1]["case_id"] = "even"

    result = unauthorized_finite_partition_services.services.core.capabilities.invoke(
        request
    )

    assert result.output["duplicate_case_ids"] == ["even"]
    assert result.completeness.status.value == "PARTIAL"
