from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.support.capabilities import invoke_capability as _invoke
from tests.support.rationals import rational_payload as _q
from tests.support.services import open_domain_services

from jacobian.bounded_process import BoundedProcessResult
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.matrices.flint_linear import install_python_flint_inconsistency_capability
from jacobian.matrices.linear_capabilities import (
    install_linear_rational_inconsistency_checker,
)
from jacobian.providers.flint_runtime import python_flint_provider_runtime
from jacobian.runtime import CheckerAuthorityMode
from jacobian.runtime.services import CoreServices


def _system(coefficients: list[list[int]], rhs: list[int]) -> dict[str, Any]:
    return {
        "variables": [f"x{index}" for index in range(len(coefficients[0]))],
        "coefficients": {
            "entries": [[_q(value) for value in row] for row in coefficients]
        },
        "rhs": [_q(value) for value in rhs],
    }


@dataclass(frozen=True, slots=True)
class _LinearRuntime:
    core: CoreServices
    provider_runtime: CapabilityProviderRuntime


@contextmanager
def _open_runtime(root: Path, *, install_checker: bool) -> Iterator[_LinearRuntime]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if install_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        runtime = python_flint_provider_runtime()
        producer = install_python_flint_inconsistency_capability(
            services.core.linear,
            runtime,
        )
        services.installation.register_capability(producer)
        if install_checker:
            adapter, _installation = install_linear_rational_inconsistency_checker(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.core.linear,
                services.installation.verification,
                services.core.checkers,
                authorize_checker=True,
            )
            assert adapter is not None
            services.installation.register_capability(adapter)
        yield _LinearRuntime(core=services.core, provider_runtime=runtime)


@pytest.fixture
def linear_services(tmp_path: Path) -> Iterator[_LinearRuntime]:
    with _open_runtime(tmp_path, install_checker=False) as services:
        yield services


@pytest.fixture
def linear_checker_services(tmp_path: Path) -> Iterator[_LinearRuntime]:
    with _open_runtime(tmp_path, install_checker=True) as services:
        yield services


def test_python_flint_finds_normalized_unverified_inconsistency_witness(
    linear_services: _LinearRuntime,
) -> None:
    runtime = linear_services
    assert (
        runtime.provider_runtime.availability
        is CapabilityProviderAvailability.AVAILABLE
    )
    result = _invoke(
        runtime,
        "linear.rational_inconsistency.find",
        {
            "system": _system([[1, 1], [2, 2]], [1, 3]),
            "resource_budget": {"wall_seconds": 5},
        },
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "CERTIFICATE_PRODUCED"
    assert result.output["left_witness"] == [_q(-2), _q(1)]
    assert result.output["rhs_pairing"] == _q(1)
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification"] == "UNVERIFIED"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert (
        result.relationships[0].relation_id
        == "linear.relation.inconsistency-certificate-of"
    )
    resolved = runtime.core.linear.resolve_inconsistency(
        result.output["certificate_uri"]
    )
    assert (
        resolved.certificate.system.system_artifact_uri == result.output["system_uri"]
    )
    assert result.output["system_uri"] in resolved.artifact.manifest.parents


def test_no_certificate_is_not_a_consistency_conclusion(
    linear_services: _LinearRuntime,
) -> None:
    runtime = linear_services
    result = _invoke(
        runtime,
        "linear.rational_inconsistency.find",
        {"system": _system([[1, 0], [0, 1]], [2, 3])},
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "NO_CERTIFICATE_PRODUCED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["certificate_uri"] is None
    assert result.completeness.status is CapabilityCompletenessStatus.UNKNOWN


def test_independent_checker_verifies_inconsistency(
    linear_checker_services: _LinearRuntime,
) -> None:
    runtime = linear_checker_services
    found = _invoke(
        runtime,
        "linear.rational_inconsistency.find",
        {"system": _system([[1, 1], [2, 2]], [1, 3])},
        mode=CapabilityMode.EXPLORE,
    )
    verified = _invoke(
        runtime,
        "linear.rational_inconsistency.verify",
        {"certificate_uri": found.output["certificate_uri"]},
        mode=CapabilityMode.VERIFY,
    )

    assert verified.output["status"] == "VERIFIED_INCONSISTENT"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.relationships[0].status.value == "VERIFIED"
    assert verified.output["verification_record_uri"].startswith("artifact://sha256/")


def test_inconsistency_timeout_retains_no_certificate(
    linear_services: _LinearRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = linear_services
    monkeypatch.setattr(
        "jacobian.matrices.flint_linear.run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )
    result = _invoke(
        runtime,
        "linear.rational_inconsistency.find",
        {"system": _system([[1]], [1])},
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["status"] == "NO_CERTIFICATE_PRODUCED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["certificate_uri"] is None
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
