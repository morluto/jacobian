from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

import jacobian_checkers.sat
from jacobian.contracts.capabilities import (
    CapabilityInputKind,
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.sat import SatResourceBudget
from jacobian.contracts.verification import VerificationRecord
from jacobian.runtime import CheckerAuthorityMode
from jacobian.sat_smt.sat_capabilities import (
    SatAssignmentCheckerInstallation,
    install_sat_assignment_checker,
)
from jacobian.verification import CheckerExecutionError


@dataclass(frozen=True, slots=True)
class SatAssignmentTestServices(DomainTestServices):
    assignment: SatAssignmentCheckerInstallation


@contextmanager
def _open_sat_assignment_services(
    root: Path,
    *,
    authorize_checker: bool,
) -> Iterator[SatAssignmentTestServices]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            adapter, installation = install_sat_assignment_checker(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.sat,
                services.application.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            if adapter is not None:
                services.installation.register_capability(adapter)
        yield SatAssignmentTestServices(
            core=services.core,
            application=services.application,
            installation=services.installation,
            assignment=installation,
        )


@pytest.fixture
def sat_assignment_services(
    tmp_path: Path,
) -> Iterator[SatAssignmentTestServices]:
    with _open_sat_assignment_services(
        tmp_path / "state", authorize_checker=True
    ) as services:
        yield services


@pytest.fixture
def unauthorized_sat_assignment_services(
    tmp_path: Path,
) -> Iterator[SatAssignmentTestServices]:
    with _open_sat_assignment_services(
        tmp_path / "state", authorize_checker=False
    ) as services:
        yield services


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
    runtime: SatAssignmentTestServices,
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


def _verify(runtime: SatAssignmentTestServices, assignment_uri: str):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.model.verify",
            input={"assignment_uri": assignment_uri},
        )
    )


def test_sat_assignment_verifier_declares_its_typed_artifact_route(
    sat_assignment_services,
) -> None:
    descriptor = next(
        descriptor
        for descriptor in sat_assignment_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "sat.model.verify"
    )

    assert CapabilityInputKind.TYPED_ARTIFACT in descriptor.accepted_input_kinds
    assert descriptor.accepted_artifact_types == (
        sat_assignment_services.core.sat.installation.assignment_schema_uri,
    )


def test_invalid_assignment_diagnostic_routes_through_public_capabilities(
    sat_assignment_services,
) -> None:
    result = _verify(
        sat_assignment_services,
        "artifact://sha256/" + "0" * 64,
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_SAT_ASSIGNMENT"
    hint = result.diagnostics[0].hint or ""
    assert "math.find" in hint
    assert "sat.model.find" in hint
    assert "CaDiCaL" in hint
    assert "SatArtifactService" not in hint


def test_missing_assignment_is_reported_as_invalid_input(
    sat_assignment_services,
) -> None:
    result = sat_assignment_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.model.verify",
            input={},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
    assert result.diagnostics[0].path == "assignment_uri"


def test_sat_assignment_is_verified_by_an_authorized_clean_process(
    sat_assignment_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cnf_uri, assignment_uri = _assignment(sat_assignment_services, values=(False, True))

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
    result = _verify(sat_assignment_services, assignment_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "VERIFIED_SATISFYING"
    assert result.output["conclusion"] == "TRUE"
    assert result.output["cnf_uri"] == cnf_uri
    assert result.output["assignment_uri"] == assignment_uri
    record_uri = result.verification_record_uri
    assert record_uri is not None
    record_artifact = sat_assignment_services.core.store.get(record_uri)
    record = VerificationRecord.model_validate(record_artifact.payload)
    assert record.checker_id == sat_assignment_services.assignment.checker_id
    assert record.evidence_uri == result.output["witness_uri"]
    assert set(record_artifact.manifest.parents) == {
        cnf_uri,
        assignment_uri,
        result.output["witness_uri"],
    }


def test_unsatisfying_assignment_is_rejected_without_an_opposite_conclusion(
    sat_assignment_services,
) -> None:
    _cnf_uri, assignment_uri = _assignment(
        sat_assignment_services, values=(False, False)
    )

    result = _verify(sat_assignment_services, assignment_uri)

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "REJECTED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None


def test_sat_assignment_verify_requires_operator_authorized_checker(
    unauthorized_sat_assignment_services,
) -> None:
    runtime = unauthorized_sat_assignment_services

    assert runtime.assignment.checker_id is None
    assert "sat.model.verify" not in {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }


def test_misbound_assignment_artifact_fails_before_checker_dispatch(
    sat_assignment_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cnf_uri, assignment_uri = _assignment(sat_assignment_services, values=(False, True))
    second = sat_assignment_services.core.sat.put_cnf(
        variable_names=("a", "b"), clauses=((1,),)
    )
    stored = sat_assignment_services.core.store.get(assignment_uri)
    payload = deepcopy(stored.payload)
    payload["cnf"]["cnf_artifact_uri"] = second.artifact_uri
    forged = sat_assignment_services.core.store.put(
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
        sat_assignment_services.application.verification,
        "_run_checker",
        unexpected_checker,
    )
    result = _verify(sat_assignment_services, forged.artifact_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.verification_record_uri is None
    assert called is False


def test_checker_timeout_cannot_create_a_sat_conclusion(
    sat_assignment_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cnf_uri, assignment_uri = _assignment(
        sat_assignment_services, values=(False, True)
    )

    def timeout(**_kwargs: Any):
        raise TimeoutError("checker execution timed out")

    monkeypatch.setattr(
        sat_assignment_services.application.verification, "_run_checker", timeout
    )
    result = _verify(sat_assignment_services, assignment_uri)

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["status"] == "TIMEOUT"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.verification_record_uri is None


def test_checker_error_cannot_create_a_sat_conclusion(
    sat_assignment_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cnf_uri, assignment_uri = _assignment(
        sat_assignment_services, values=(False, True)
    )

    def fail(**_kwargs: Any):
        raise CheckerExecutionError("deliberate checker failure")

    monkeypatch.setattr(
        sat_assignment_services.application.verification, "_run_checker", fail
    )
    result = _verify(sat_assignment_services, assignment_uri)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["status"] == "ERROR"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.verification_record_uri is None
