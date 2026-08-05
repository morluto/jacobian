from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

import jacobian_checkers.sat
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityInputKind,
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.sat import SatResourceBudget
from jacobian.contracts.verification import VerificationRecord
from jacobian.runtime.model import JacobianRuntime
from jacobian.verification import CheckerExecutionError


def _producer() -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="cadical",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="2.1.3",
        digest="sha256:" + "d" * 64,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T2,
        license_id="MIT",
    )


def _assignment(
    runtime: JacobianRuntime,
    *,
    values: tuple[bool, bool],
) -> tuple[str, str]:
    cnf = runtime.core.sat.put_cnf(
        variable_names=("a", "b"),
        clauses=((-1, 2), (1, 2)),
    )
    assignment = runtime.core.sat.put_assignment(
        cnf_uri=cnf.artifact_uri,
        values=values,
        producer=_producer(),
        resource_budget=SatResourceBudget(wall_seconds=30),
    )
    return cnf.artifact_uri, assignment.artifact_uri


def _verify(runtime: JacobianRuntime, assignment_uri: str):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.model.verify",
            mode=CapabilityMode.VERIFY,
            input={"assignment_uri": assignment_uri},
        )
    )


def test_sat_assignment_verifier_declares_its_typed_artifact_route(
    authorized_complete_runtime,
) -> None:
    descriptor = next(
        descriptor
        for descriptor in authorized_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "sat.model.verify"
    )

    assert CapabilityInputKind.TYPED_ARTIFACT in descriptor.accepted_input_kinds
    assert descriptor.accepted_artifact_types == (
        authorized_complete_runtime.core.sat.installation.assignment_schema_uri,
    )


def test_sat_assignment_is_verified_by_an_authorized_clean_process(
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cnf_uri, assignment_uri = _assignment(
        authorized_complete_runtime, values=(False, True)
    )

    monkeypatch.setattr(
        jacobian_checkers.sat,
        "check_assignment",
        lambda _request: {
            "accepted": False,
            "conclusion": "UNKNOWN",
            "arithmetic": "SYMBOLIC",
            "method": "DIRECT_WITNESS",
            "coverage": "NOT_APPLICABLE",
            "detail": "parent-process monkeypatch",
        },
    )
    result = _verify(authorized_complete_runtime, assignment_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.output["status"] == "VERIFIED_SATISFYING"
    assert result.output["conclusion"] == "TRUE"
    assert result.output["cnf_uri"] == cnf_uri
    assert result.output["assignment_uri"] == assignment_uri
    record_uri = result.assurance.verification_record_uri
    assert record_uri is not None
    record_artifact = authorized_complete_runtime.core.store.get(record_uri)
    record = VerificationRecord.model_validate(record_artifact.payload)
    assert (
        record.checker_id
        == authorized_complete_runtime.portfolio.sat_assignment_checker.checker_id
    )
    assert record.evidence_uri == result.output["witness_uri"]
    assert set(record_artifact.manifest.parents) == {
        cnf_uri,
        assignment_uri,
        result.output["witness_uri"],
    }


def test_unsatisfying_assignment_is_rejected_without_an_opposite_conclusion(
    authorized_complete_runtime,
) -> None:
    _cnf_uri, assignment_uri = _assignment(
        authorized_complete_runtime, values=(False, False)
    )

    result = _verify(authorized_complete_runtime, assignment_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.output["status"] == "REJECTED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None
    assert result.assurance.verification_record_uri is None


def test_sat_assignment_verify_requires_operator_authorized_checker(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime

    assert runtime.portfolio.sat_assignment_checker.checker_id is None
    assert "sat.model.verify" not in {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }


def test_misbound_assignment_artifact_fails_before_checker_dispatch(
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cnf_uri, assignment_uri = _assignment(
        authorized_complete_runtime, values=(False, True)
    )
    second = authorized_complete_runtime.core.sat.put_cnf(
        variable_names=("a", "b"), clauses=((1,),)
    )
    stored = authorized_complete_runtime.core.store.get(assignment_uri)
    payload = deepcopy(stored.payload)
    payload["cnf"]["cnf_artifact_uri"] = second.artifact_uri
    forged = authorized_complete_runtime.core.store.put(
        schema_uri=stored.manifest.schema_uri,
        semantics_uri=stored.manifest.semantics_uri,
        payload=payload,
        parents=(cnf_uri,),
        summary="forged SAT assignment binding",
    )
    called = False

    def unexpected_checker(**_kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("checker must not receive malformed source bindings")

    monkeypatch.setattr(
        authorized_complete_runtime.services.verification,
        "_run_checker",
        unexpected_checker,
    )
    result = _verify(authorized_complete_runtime, forged.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
    assert called is False


def test_checker_timeout_cannot_create_a_sat_conclusion(
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cnf_uri, assignment_uri = _assignment(
        authorized_complete_runtime, values=(False, True)
    )

    def timeout(**_kwargs: Any):
        raise TimeoutError("checker execution timed out")

    monkeypatch.setattr(
        authorized_complete_runtime.services.verification, "_run_checker", timeout
    )
    result = _verify(authorized_complete_runtime, assignment_uri)

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
    assert result.output["status"] == "TIMEOUT"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.assurance.verification_record_uri is None


def test_checker_error_cannot_create_a_sat_conclusion(
    authorized_complete_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cnf_uri, assignment_uri = _assignment(
        authorized_complete_runtime, values=(False, True)
    )

    def fail(**_kwargs: Any):
        raise CheckerExecutionError("deliberate checker failure")

    monkeypatch.setattr(
        authorized_complete_runtime.services.verification, "_run_checker", fail
    )
    result = _verify(authorized_complete_runtime, assignment_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
    assert result.output["status"] == "ERROR"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.assurance.verification_record_uri is None
