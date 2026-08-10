from __future__ import annotations

from typing import Any

import pytest
from tests.support.polynomials import univariate_term as _term

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRelationshipStatus,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.verification import CheckerExecutionError


def _input(value: int) -> dict[str, Any]:
    return {
        "system": {
            "system_schema_version": "1",
            "domain": "QQ",
            "variables": ["x"],
            "equations": [{"terms": [_term(1, 2), _term(-4, 0)]}],
            "inequations": [{"terms": [_term(1, 1)]}],
        },
        "assignment": [{"num": str(value), "den": "1"}],
    }


def test_solution_capability_verifies_valid_assignment(
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            input=_input(2),
        )
    )

    assert result.output["satisfies"] is True
    assert result.output["equation_residuals"] == [{"num": "0", "den": "1"}]
    assert result.output["inequation_values"] == [{"num": "2", "den": "1"}]
    assert result.output["residuals_assurance"] == "VERIFIED"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.relationships[0].relation_id == (
        "polynomial.relation.satisfies-system"
    )
    assert result.relationships[0].status is CapabilityRelationshipStatus.VERIFIED
    assert (
        result.relationships[0].verification_record_uri
        == result.assurance.verification_record_uri
    )
    certificate = authorized_complete_runtime.core.store.get(
        result.output["certificate_uri"]
    )
    assert (
        certificate.payload["payload"]["equation_residuals"]
        == (result.output["equation_residuals"])
    )
    record = authorized_complete_runtime.core.store.get(
        result.output["verification_record_uri"]
    )
    assert result.output["certificate_uri"] in record.manifest.parents
    assert record.payload["relationship_source_artifact_uris"] == [
        result.output["assignment_uri"]
    ]
    assert record.payload["relationship_target_artifact_uris"] == [
        result.output["system_uri"]
    ]
    assert record.payload["obligation_uri"] is None


def test_solution_capability_verifies_invalid_assignment(
    authorized_complete_runtime,
) -> None:

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            input=_input(1),
        )
    )

    assert result.output["satisfies"] is False
    assert result.output["conclusion"] == "FALSE"
    assert result.output["residuals_assurance"] == "VERIFIED"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.relationships == ()
    record = authorized_complete_runtime.core.store.get(
        result.output["verification_record_uri"]
    )
    assert record.payload["relation_id"] is None
    assert record.payload["relationship_source_artifact_uris"] == []
    assert record.payload["relationship_target_artifact_uris"] == []
    assert record.payload["obligation_uri"] is None


def test_solution_capability_keeps_checker_failure_unknown(
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def fail(**_kwargs: Any):
        raise CheckerExecutionError("deliberate checker failure")

    monkeypatch.setattr(
        authorized_complete_runtime.services.verification, "_run_checker", fail
    )
    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            input=_input(1),
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["satisfies"] is None
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["residuals_assurance"] == "COMPUTED"
    assert result.output["verification_record_uri"] is None
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
    assert result.relationships == ()


def test_solution_capability_is_only_available_with_checker(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime

    ids = {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }

    assert "polynomial.system.solution.verify" not in ids
