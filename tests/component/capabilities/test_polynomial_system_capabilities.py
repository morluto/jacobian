from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from tests.support.polynomials import univariate_term as _term
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.polynomial_system_capabilities import (
    install_polynomial_system_capabilities,
)
from jacobian.runtime import CheckerAuthorityMode
from jacobian.verification import CheckerExecutionError


@contextmanager
def _open_polynomial_system_services(
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
            adapter, _installation = install_polynomial_system_capabilities(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.application.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorizes_bundled_checkers,
            )
            if adapter is not None:
                services.installation.register_capability(adapter)
        yield services


@pytest.fixture
def polynomial_system_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with _open_polynomial_system_services(
        tmp_path / "state", authorize_checker=True
    ) as services:
        yield services


@pytest.fixture
def unauthorized_polynomial_system_services(
    tmp_path: Path,
) -> Iterator[DomainTestServices]:
    with _open_polynomial_system_services(
        tmp_path / "state", authorize_checker=False
    ) as services:
        yield services


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
    polynomial_system_services,
) -> None:

    result = polynomial_system_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            input=_input(2),
        )
    )

    assert result.output["satisfies"] is True
    assert result.output["equation_residuals"] == [{"num": "0", "den": "1"}]
    assert result.output["inequation_values"] == [{"num": "2", "den": "1"}]
    assert result.verification_record_uri is not None
    certificate = polynomial_system_services.core.store.get(
        result.output["certificate_uri"]
    )
    assert (
        certificate.payload["payload"]["equation_residuals"]
        == (result.output["equation_residuals"])
    )
    record = polynomial_system_services.core.store.get(
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
    polynomial_system_services,
) -> None:

    result = polynomial_system_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            input=_input(1),
        )
    )

    assert result.output["satisfies"] is False
    assert result.output["conclusion"] == "FALSE"
    assert result.verification_record_uri is not None
    record = polynomial_system_services.core.store.get(
        result.output["verification_record_uri"]
    )
    assert record.payload["relation_id"] is None
    assert record.payload["relationship_source_artifact_uris"] == []
    assert record.payload["relationship_target_artifact_uris"] == []
    assert record.payload["obligation_uri"] is None


def test_solution_capability_keeps_checker_failure_unknown(
    polynomial_system_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def fail(**_kwargs: Any):
        raise CheckerExecutionError("deliberate checker failure")

    monkeypatch.setattr(
        polynomial_system_services.application.verification, "_run_checker", fail
    )
    result = polynomial_system_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            input=_input(1),
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["satisfies"] is None
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None


def test_solution_capability_rejects_dimension_mismatch_before_artifact_writes(
    polynomial_system_services: DomainTestServices,
) -> None:
    connection = sqlite3.connect(polynomial_system_services.core.store.db_path)
    try:
        before = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        connection.close()
    invalid = _input(2)
    invalid["assignment"].append({"num": "3", "den": "1"})

    result = polynomial_system_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="polynomial.system.solution.verify",
            input=invalid,
        )
    )

    connection = sqlite3.connect(polynomial_system_services.core.store.db_path)
    try:
        after = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        connection.close()
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_POLYNOMIAL_SYSTEM_SOLUTION_REQUEST"
    assert result.diagnostics[0].stage == "request_validation"
    assert before == after


def test_solution_capability_is_only_available_with_checker(
    unauthorized_polynomial_system_services,
) -> None:
    runtime = unauthorized_polynomial_system_services

    ids = {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }

    assert "polynomial.system.solution.verify" not in ids
