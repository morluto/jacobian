from __future__ import annotations

from tests.support.core_capability_harnesses import FiniteCoverageTestServices

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityObligationStatus,
    CapabilityRelationshipStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus


def _request(
    scope: list[str | int],
    pages: list[list[str | int]],
    *,
    canonicalizer_id: str = "finite.string.nfc@1",
) -> CapabilityRequest:
    return CapabilityRequest(
        capability_id="finite.coverage.verify",
        input={
            "canonicalizer_id": canonicalizer_id,
            "scope_items": scope,
            "pages": [{"items": items} for items in pages],
        },
    )


def test_finite_coverage_verifies_exactly_once_across_pages(
    finite_coverage_services: FiniteCoverageTestServices,
) -> None:
    runtime = finite_coverage_services.services
    result = runtime.core.capabilities.invoke(
        _request(["alpha", "beta", "gamma"], [["alpha"], ["beta", "gamma"]])
    )

    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.output["coverage_status"] == "EXACTLY_ONCE"
    assert result.output["conclusion"] == "TRUE"
    assert result.output["diagnostics"] == {
        "missing_keys": [],
        "duplicate_keys": [],
        "outside_keys": [],
        "duplicate_occurrences": [],
    }
    assert len(result.output["page_uris"]) == 2
    assert result.output["verification_record_uri"] is not None
    assert result.relationships[0].status is CapabilityRelationshipStatus.VERIFIED
    assert result.obligations[0].status is CapabilityObligationStatus.DISCHARGED

    archive = runtime.core.store.get(result.output["archive_uri"])
    assert set(result.output["page_uris"]).issubset(set(archive.manifest.parents))
    assert result.output["scope_uri"] in archive.manifest.parents
    assert result.output["canonicalizer_uri"] in archive.manifest.parents


def test_finite_coverage_reports_omission_and_duplicate(
    finite_coverage_services: FiniteCoverageTestServices,
) -> None:
    runtime = finite_coverage_services.services
    result = runtime.core.capabilities.invoke(
        _request(["alpha", "beta", "gamma"], [["alpha", "beta"], ["beta"]])
    )

    diagnostics = result.output["diagnostics"]
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.completeness.status is CapabilityCompletenessStatus.PARTIAL
    assert result.output["coverage_status"] == "INVALID"
    assert result.output["conclusion"] == "UNKNOWN"
    assert len(diagnostics["missing_keys"]) == 1
    assert len(diagnostics["duplicate_keys"]) == 1
    assert len(diagnostics["duplicate_occurrences"]) == 2
    assert result.output["verification_record_uri"] is None
    assert result.relationships[0].status is CapabilityRelationshipStatus.PROPOSED
    assert result.obligations[0].status is CapabilityObligationStatus.OPEN


def test_finite_coverage_reports_items_outside_scope(
    finite_coverage_services: FiniteCoverageTestServices,
) -> None:
    runtime = finite_coverage_services.services
    result = runtime.core.capabilities.invoke(
        _request(["alpha", "beta"], [["alpha", "beta", "gamma"]])
    )

    assert result.output["coverage_status"] == "INVALID"
    assert len(result.output["diagnostics"]["outside_keys"]) == 1
    assert result.output["verification_record_uri"] is None


def test_finite_coverage_supports_registered_integer_canonicalizer(
    finite_coverage_services: FiniteCoverageTestServices,
) -> None:
    runtime = finite_coverage_services.services
    result = runtime.core.capabilities.invoke(
        _request(
            [1, 2, 3],
            [[3], [1, 2]],
            canonicalizer_id="finite.integer.decimal@1",
        )
    )

    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.output["canonicalizer_id"] == "finite.integer.decimal@1"


def test_finite_coverage_unknown_canonicalizer_reports_precise_schema_error(
    finite_coverage_services: FiniteCoverageTestServices,
) -> None:
    runtime = finite_coverage_services.services
    result = runtime.core.capabilities.invoke(
        _request(
            ["alpha"],
            [["alpha"]],
            canonicalizer_id="finite.unknown@1",
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "INVALID_REQUEST"
    assert diagnostic.path == "canonicalizer_id"
    assert diagnostic.actual_type == "string"
    assert diagnostic.expected == (
        'one of: "finite.integer.decimal@1", "finite.string.nfc@1"'
    )


def test_finite_coverage_rejects_nfc_collisions_in_scope(
    finite_coverage_services: FiniteCoverageTestServices,
) -> None:
    runtime = finite_coverage_services.services
    result = runtime.core.capabilities.invoke(_request(["é", "e\u0301"], [["é"]]))

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["error"]["code"] == "DUPLICATE_FINITE_SCOPE_KEY"


def test_finite_coverage_is_unavailable_without_authorized_checker(
    unauthorized_finite_coverage_services: FiniteCoverageTestServices,
) -> None:
    result = unauthorized_finite_coverage_services.services.core.capabilities.invoke(
        _request(["alpha"], [["alpha"]])
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["error"]["code"] == "UNKNOWN_CAPABILITY"
