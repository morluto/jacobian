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

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime.config import CheckerAuthorityMode
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
    assert result.output["verification_record_uri"] is not None
    proof = lrat_services.core.store.get(result.output["proof_uri"])
    assert proof.manifest.parents == (cnf.artifact_uri,)


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


def test_rejected_lrat_reports_the_first_invalid_proof_step(
    lrat_services,
) -> None:
    cnf = lrat_services.core.sat.put_cnf(variable_names=("x",), clauses=((-1,), (1,)))

    result = _verify(lrat_services, cnf.artifact_uri, b"3 0 1 99 0\n")

    assert result.output["status"] == "REJECTED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["invalid_step"] == {
        "line": 1,
        "clause_id": 3,
        "code": "HINT_REFERENCES_INACTIVE_CLAUSE",
        "proof_line": "3 0 1 99 0",
        "proof_line_truncated": False,
        "raw_checker_message": "line 1: hint references inactive clause",
    }


def test_rejected_lrat_bounds_an_oversized_invalid_proof_line(
    lrat_services,
) -> None:
    cnf = lrat_services.core.sat.put_cnf(variable_names=("x",), clauses=((-1,), (1,)))
    oversized_line = "invalid-clause-id " + "1 " * 2500

    result = _verify(
        lrat_services,
        cnf.artifact_uri,
        (oversized_line + "\n").encode("ascii"),
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "REJECTED"
    assert result.output["conclusion"] == "UNKNOWN"
    invalid_step = result.output["invalid_step"]
    assert invalid_step["code"] == "NON_INTEGER_TOKEN"
    assert len(invalid_step["proof_line"]) == 4096
    assert invalid_step["proof_line"] == oversized_line[:4096]
    assert invalid_step["proof_line_truncated"] is True


def test_timeout_and_cancellation_are_fail_closed(lrat_services) -> None:
    cnf = lrat_services.core.sat.put_cnf(variable_names=("x",), clauses=((-1,), (1,)))

    timed_out = _verify(
        lrat_services,
        cnf.artifact_uri,
        b"3 0 1 2 0\n",
        limits={"timeout_ms": 0},
    )
    cancelled = _verify(lrat_services, cnf.artifact_uri, b"3 0 1 2 0\n", cancelled=True)

    assert timed_out.output == {}
    assert timed_out.execution.status is ExecutionStatus.TIMEOUT
    assert cancelled.output == {}
    assert cancelled.execution.status is ExecutionStatus.CANCELLED


@pytest.mark.parametrize(
    ("proof", "limits"),
    (
        (b"3 0 1 2 0\n4 0 1 2 0\n", {"max_steps": 1}),
        (b"3 1 0 1 2 0\n", {"max_clause_literals": 0}),
    ),
)
def test_lrat_resource_exhaustion_is_an_operational_error(
    lrat_services,
    proof: bytes,
    limits: dict[str, int],
) -> None:
    cnf = lrat_services.core.sat.put_cnf(variable_names=("x",), clauses=((-1,), (1,)))

    result = _verify(lrat_services, cnf.artifact_uri, proof, limits=limits)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output == {}
    assert result.verification_record_uri is None


def test_capability_is_absent_without_operator_authorized_references(
    unauthorized_lrat_services: DomainTestServices,
) -> None:
    runtime = unauthorized_lrat_services
    ids = {
        item.capability_id for item in runtime.core.capabilities.catalog().capabilities
    }
    assert "sat.lrat.verify" not in ids
