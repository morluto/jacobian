from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime import CheckerAuthorityMode
from jacobian.sat_smt.sat_lrat import install_sat_lrat_verifier


@contextmanager
def _open_lrat_services(
    root: Path,
    *,
    authorize_checker: bool,
) -> Iterator[DomainTestServices]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if authorize_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        with atomic_installation(services.core):
            adapter, _installation = install_sat_lrat_verifier(
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
        yield services


@pytest.fixture
def lrat_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with _open_lrat_services(tmp_path / "state", authorize_checker=True) as services:
        yield services


@pytest.fixture
def unauthorized_lrat_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with _open_lrat_services(tmp_path / "state", authorize_checker=False) as services:
        yield services


def _verify(runtime: DomainTestServices, cnf_uri: str, proof: bytes, **extra: object):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="sat.lrat.verify",
            input={
                "cnf_uri": cnf_uri,
                "proof_base64": base64.b64encode(proof).decode("ascii"),
                **extra,
            },
        )
    )


def test_rup_lrat_derives_empty_clause_and_binds_artifacts(
    lrat_services,
) -> None:
    cnf = lrat_services.core.sat.put_cnf(variable_names=("x",), clauses=((-1,), (1,)))

    result = _verify(lrat_services, cnf.artifact_uri, b"3 0 1 2 0\n")

    assert result.output["status"] == "VERIFIED_UNSAT"
    assert result.output["conclusion"] == "TRUE"
    assert result.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert result.output["verification_record_uri"] is not None
    proof = lrat_services.core.store.get(result.output["proof_uri"])
    assert proof.manifest.parents == (cnf.artifact_uri,)
    assert (
        proof.payload["cnf"]["variable_map_digest"]
        == result.scope.parameters["variable_map_digest"]
    )
    assert (
        proof.payload["cnf"]["dimacs_digest"]
        == result.scope.parameters["dimacs_digest"]
    )


def test_invalid_or_incomplete_lrat_never_proves_sat_or_unsat(
    lrat_services,
) -> None:
    for proof in (
        b"3 0 1 0\n",  # no conflict
        b"3 0 1 99 0\n",  # missing hint
        b"3 1 0 1 2 0\n",  # only a nonempty derived clause
        b"3 0 1 2",  # truncated framing
    ):
        cnf = lrat_services.core.sat.put_cnf(
            variable_names=("x",), clauses=((-1,), (1,))
        )

        result = _verify(lrat_services, cnf.artifact_uri, proof)

        assert result.output["status"] == "REJECTED", proof
        assert result.output["conclusion"] == "UNKNOWN", proof
        assert result.output["verification_record_uri"] is None, proof


def test_negative_rat_hint_is_explicitly_unsupported(
    lrat_services,
) -> None:
    cnf = lrat_services.core.sat.put_cnf(variable_names=("x",), clauses=((-1,), (1,)))

    result = _verify(lrat_services, cnf.artifact_uri, b"3 0 -1 2 0\n")

    assert result.output["status"] == "UNSUPPORTED"
    assert result.output["conclusion"] == "UNKNOWN"


def test_timeout_and_cancellation_are_fail_closed(lrat_services) -> None:
    cnf = lrat_services.core.sat.put_cnf(variable_names=("x",), clauses=((-1,), (1,)))

    timed_out = _verify(
        lrat_services,
        cnf.artifact_uri,
        b"3 0 1 2 0\n",
        limits={"timeout_ms": 0},
    )
    cancelled = _verify(lrat_services, cnf.artifact_uri, b"3 0 1 2 0\n", cancelled=True)

    assert timed_out.output["status"] == "TIMEOUT"
    assert timed_out.output["conclusion"] == "UNKNOWN"
    assert timed_out.execution.status is ExecutionStatus.TIMEOUT
    assert cancelled.output["status"] == "CANCELLED"
    assert cancelled.output["conclusion"] == "UNKNOWN"
    assert cancelled.execution.status is ExecutionStatus.CANCELLED


def test_capability_is_absent_without_operator_authorized_references(
    unauthorized_lrat_services: DomainTestServices,
) -> None:
    runtime = unauthorized_lrat_services
    ids = {
        item.capability_id for item in runtime.core.capabilities.catalog().capabilities
    }
    assert "sat.lrat.verify" not in ids
