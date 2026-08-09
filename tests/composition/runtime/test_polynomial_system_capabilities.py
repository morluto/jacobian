from __future__ import annotations

import sqlite3
import subprocess
import sys
from typing import Any

import pytest
from tests.support.polynomials import univariate_term as _term

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
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
            mode=CapabilityMode.VERIFY,
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
            mode=CapabilityMode.VERIFY,
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
            mode=CapabilityMode.VERIFY,
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


def test_solution_capability_rejects_dimension_mismatch_before_artifact_writes(
    authorized_complete_runtime,
) -> None:
    connection = sqlite3.connect(authorized_complete_runtime.core.store.db_path)
    try:
        before = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        connection.close()
    invalid = _input(2)
    invalid["assignment"].append({"num": "3", "den": "1"})

    result = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            mode=CapabilityMode.VERIFY,
            input=invalid,
        )
    )

    connection = sqlite3.connect(authorized_complete_runtime.core.store.db_path)
    try:
        after = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        connection.close()
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_POLYNOMIAL_SYSTEM_SOLUTION_REQUEST"
    assert result.diagnostics[0].stage == "request_validation"
    assert before == after


def test_solution_capability_is_only_available_with_checker(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime

    ids = {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }

    assert "polynomial.system.solution.verify" not in ids


def test_solution_adapter_rejects_missing_checker_under_optimized_python() -> None:
    """An optimized interpreter must not erase checker-authorization guards."""

    script = """
from jacobian.polynomial_system_capabilities import (
    PolynomialSystemInstallation,
    PolynomialSystemResources,
    PolynomialSystemSolutionAdapter,
)

installation = PolynomialSystemInstallation(
    semantics_uri="semantics://test",
    system_schema_uri="schema://system",
    assignment_schema_uri="schema://assignment",
    claim_schema_uri="schema://claim",
    certificate_schema_uri="schema://certificate",
    checker_id=None,
)
resources = PolynomialSystemResources(
    store=None,
    artifacts=None,
    verification=None,
    installation=installation,
)
try:
    PolynomialSystemSolutionAdapter(resources)
except RuntimeError as exc:
    if "authorized checker" not in str(exc):
        raise
else:
    raise SystemExit("missing checker did not prevent adapter construction")
"""
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
