from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.support.capabilities import invoke_capability as _invoke
from tests.support.rationals import rational_payload as _q
from tests.support.services import open_domain_services

import jacobian.providers.flint_runtime as flint_runtime
from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityInstallTier,
    CapabilityMode,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.matrices.flint_linear import install_python_flint_linear_capability
from jacobian.matrices.linear_capabilities import (
    install_linear_rational_solution_checker,
)
from jacobian.process_policy import ProcessResult, ProcessTermination
from jacobian.provider_runtime import PYTHON_FLINT_VERSION
from jacobian.providers.flint_runtime import python_flint_provider_runtime
from jacobian.runtime import CheckerAuthorityMode, create_runtime
from jacobian.runtime.services import CoreServices


def _system(
    coefficients: list[list[tuple[int, int] | int]],
    rhs: list[tuple[int, int] | int],
) -> dict[str, Any]:
    def wire(value: tuple[int, int] | int) -> dict[str, str]:
        return _q(*value) if isinstance(value, tuple) else _q(value)

    return {
        "variables": [f"x{index}" for index in range(len(coefficients[0]))],
        "coefficients": {
            "entries": [[wire(value) for value in row] for row in coefficients]
        },
        "rhs": [wire(value) for value in rhs],
    }


@dataclass(frozen=True, slots=True)
class _LinearRuntime:
    core: CoreServices
    provider_runtime: CapabilityProviderRuntime


@contextmanager
def _open_linear_runtime(
    root: Path,
    *,
    install_checker: bool,
) -> Iterator[_LinearRuntime]:
    authority = (
        CheckerAuthorityMode.INSTALL_BUNDLED
        if install_checker
        else CheckerAuthorityMode.NONE
    )
    with open_domain_services(root, checker_authority=authority) as services:
        runtime = python_flint_provider_runtime()
        producer = install_python_flint_linear_capability(
            services.core.linear,
            runtime,
        )
        services.installation.register_capability(producer)
        if install_checker:
            adapter, _installation = install_linear_rational_solution_checker(
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
    with _open_linear_runtime(tmp_path, install_checker=False) as services:
        yield services


@pytest.fixture
def linear_checker_services(tmp_path: Path) -> Iterator[_LinearRuntime]:
    with _open_linear_runtime(tmp_path, install_checker=True) as services:
        yield services


def test_python_flint_find_returns_one_exact_unverified_solution(
    linear_services: _LinearRuntime,
) -> None:
    runtime = linear_services
    assert (
        runtime.provider_runtime.availability
        is CapabilityProviderAvailability.AVAILABLE
    )
    result = _invoke(
        runtime,
        "linear.rational_solution.find",
        {
            "system": _system([[2, 1], [1, -1]], [5, 1]),
            "resource_budget": {"wall_seconds": 5},
        },
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "SOLUTION_PRODUCED"
    assert result.output["solution"] == [_q(2), _q(1)]
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification"] == "UNVERIFIED"
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.relationships[0].relation_id == "linear.relation.satisfies"

    resolved = runtime.core.linear.resolve_solution(result.output["solution_uri"])
    assert resolved.solution.system.system_artifact_uri == result.output["system_uri"]
    assert result.output["system_uri"] in resolved.artifact.manifest.parents


def test_python_flint_runtime_is_exact_optional_distribution_identity() -> None:
    runtime = python_flint_provider_runtime()

    assert runtime.version == PYTHON_FLINT_VERSION
    assert runtime.install_tier is CapabilityInstallTier.T1
    assert runtime.digest.startswith("sha256:")
    assert runtime.digest_kind.value == "PYTHON_DISTRIBUTION_RECORD"
    assert runtime.license_id == "MIT AND LGPL-3.0-or-later"
    assert runtime.license_files == ("python_flint-0.9.0.dist-info/licenses/LICENSE",)
    assert runtime.configuration["operation"] == "fmpq_mat.rref"


def test_python_flint_runtime_rejects_an_unpinned_binding_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong = CapabilityProviderRuntime(
        provider="python-flint",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="0.8.0",
        digest="sha256:" + "d" * 64,
        digest_kind=CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        license_files=("python_flint-0.8.0.dist-info/licenses/LICENSE",),
    )
    monkeypatch.setattr(
        flint_runtime,
        "python_distribution_provider_runtime",
        lambda *_args, **_kwargs: wrong,
    )

    rejected = python_flint_provider_runtime()

    assert rejected.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert rejected.version is None
    assert rejected.digest is None
    assert "pinned 0.9.0" in rejected.diagnostic


def test_verifier_authorization_is_separate_from_provider_availability(
    tmp_path: Path,
    complete_portfolio_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = CapabilityProviderRuntime(
        provider="python-flint",
        availability=CapabilityProviderAvailability.UNAVAILABLE,
        platform="linux-x86_64",
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        diagnostic="optional provider unavailable for test",
    )
    monkeypatch.setattr(
        "jacobian.portfolio.provider_resolution.python_flint_provider_runtime",
        lambda: unavailable,
    )

    default_root = tmp_path / "default"
    authorized_root = tmp_path / "authorized"
    shutil.copytree(complete_portfolio_template, default_root)
    shutil.copytree(complete_portfolio_template, authorized_root)
    with create_runtime(default_root) as default:
        default_ids = {
            descriptor.capability_id
            for descriptor in default.core.capabilities.catalog().capabilities
        }
    assert "linear.rational_solution.find" not in default_ids
    assert "linear.rational_solution.verify" not in default_ids

    with create_runtime(
        authorized_root,
        checker_authority=CheckerAuthorityMode.INSTALL_BUNDLED,
    ) as authorized:
        authorized_ids = {
            descriptor.capability_id
            for descriptor in authorized.core.capabilities.catalog().capabilities
        }
    assert "linear.rational_solution.find" not in authorized_ids
    assert "linear.rational_solution.verify" in authorized_ids


def test_python_flint_find_handles_underdetermined_system_deterministically(
    linear_services: _LinearRuntime,
) -> None:
    runtime = linear_services
    result = _invoke(
        runtime,
        "linear.rational_solution.find",
        {
            "system": _system([[1, 1, 1], [2, -1, 1]], [3, 2]),
            "resource_budget": {"wall_seconds": 5},
        },
        mode=CapabilityMode.EXPLORE,
    )

    assert result.output["status"] == "SOLUTION_PRODUCED"
    assert result.output["solution"] == [_q(5, 3), _q(4, 3), _q(0)]


def test_not_found_is_not_an_inconsistency_conclusion(
    linear_services: _LinearRuntime,
) -> None:
    runtime = linear_services
    result = _invoke(
        runtime,
        "linear.rational_solution.find",
        {
            "system": _system([[1, 1], [1, 1]], [2, 3]),
            "resource_budget": {"wall_seconds": 5},
        },
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "NO_SOLUTION_PRODUCED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["solution_uri"] is None
    assert result.completeness.status is CapabilityCompletenessStatus.UNKNOWN
    assert "inconsistent" not in result.output["detail"].lower()


def test_independent_checker_verifies_and_rejects_bound_solutions(
    linear_checker_services: _LinearRuntime,
) -> None:
    runtime = linear_checker_services
    found = _invoke(
        runtime,
        "linear.rational_solution.find",
        {
            "system": _system([[2, 1], [1, -1]], [5, 1]),
            "resource_budget": {"wall_seconds": 5},
        },
        mode=CapabilityMode.EXPLORE,
    )
    accepted = _invoke(
        runtime,
        "linear.rational_solution.verify",
        {"solution_uri": found.output["solution_uri"]},
        mode=CapabilityMode.VERIFY,
    )
    assert accepted.output["status"] == "VERIFIED_SOLUTION"
    assert accepted.output["conclusion"] == "TRUE"
    assert accepted.output["verification_record_uri"].startswith("artifact://sha256/")
    assert accepted.assurance.level is CapabilityAssuranceLevel.VERIFIED

    wrong = runtime.core.linear.put_solution(
        system_uri=found.output["system_uri"],
        values=(_q(0), _q(0)),
        producer=runtime.provider_runtime,
        resource_budget={"wall_seconds": 5},
    )
    rejected = _invoke(
        runtime,
        "linear.rational_solution.verify",
        {"solution_uri": wrong.artifact_uri},
        mode=CapabilityMode.VERIFY,
    )
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_python_flint_timeout_is_operational_not_mathematical(
    linear_services: _LinearRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = linear_services
    monkeypatch.setattr(
        "jacobian.matrices.flint_linear.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )
    result = _invoke(
        runtime,
        "linear.rational_solution.find",
        {
            "system": _system([[1]], [1]),
            "resource_budget": {"wall_seconds": 1},
        },
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["status"] == "NO_SOLUTION_PRODUCED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED


def test_python_flint_worker_gets_only_fixed_environment_and_exact_budget(
    linear_services: _LinearRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = linear_services
    monkeypatch.setenv("JACOBIAN_LINEAR_SECRET", "must-not-propagate")
    observed: dict[str, Any] = {}

    def fake_worker(request: Any) -> ProcessResult:
        observed["timeout_seconds"] = request.timeout_seconds
        observed["environment"] = dict(request.environment)
        stdout = (
            canonicalize_json(
                {
                    "protocol": "jacobian.flint-linear-worker/v1",
                    "status": "SOLUTION_PRODUCED",
                    "backend_version": "0.9.0",
                    "values": [_q(1)],
                }
            )
            + b"\n"
        )
        return ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=stdout,
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr("jacobian.matrices.flint_linear.execute_process", fake_worker)
    result = _invoke(
        runtime,
        "linear.rational_solution.find",
        {
            "system": _system([[1]], [1]),
            "resource_budget": {"wall_seconds": 7},
        },
        mode=CapabilityMode.EXPLORE,
    )

    assert result.output["status"] == "SOLUTION_PRODUCED"
    assert observed["timeout_seconds"] == 7.0
    assert observed["environment"] == {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert "JACOBIAN_LINEAR_SECRET" in os.environ
    assert "JACOBIAN_LINEAR_SECRET" not in observed["environment"]


def test_python_flint_discards_output_if_runtime_identity_changes(
    linear_services: _LinearRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = linear_services
    original_runtime = runtime.provider_runtime
    changed_runtime = original_runtime.model_copy(
        update={"digest": "sha256:" + "f" * 64}
    )
    observations = iter((original_runtime, changed_runtime))
    monkeypatch.setattr(
        "jacobian.matrices.flint_linear.python_flint_provider_runtime",
        lambda **_kwargs: next(observations),
    )
    monkeypatch.setattr(
        "jacobian.matrices.flint_linear.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=(
                canonicalize_json(
                    {
                        "protocol": "jacobian.flint-linear-worker/v1",
                        "status": "SOLUTION_PRODUCED",
                        "backend_version": "0.9.0",
                        "values": [_q(1)],
                    }
                )
                + b"\n"
            ),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    result = _invoke(
        runtime,
        "linear.rational_solution.find",
        {"system": _system([[1]], [1])},
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["status"] == "NO_SOLUTION_PRODUCED"
    assert result.output["solution_uri"] is None
    assert result.output["conclusion"] == "UNKNOWN"
    assert "changed during execution" in result.output["detail"]


def test_invalid_worker_protocol_fails_without_solution_evidence(
    linear_services: _LinearRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = linear_services
    monkeypatch.setattr(
        "jacobian.matrices.flint_linear.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=b'{"status":"SOLUTION_PRODUCED","values":[]}\n',
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )
    result = _invoke(
        runtime,
        "linear.rational_solution.find",
        {"system": _system([[1]], [1])},
        mode=CapabilityMode.EXPLORE,
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["status"] == "NO_SOLUTION_PRODUCED"
    assert result.output["solution_uri"] is None
    assert result.output["conclusion"] == "UNKNOWN"


def test_linear_checker_timeout_is_operational_not_mathematical(
    linear_checker_services: _LinearRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = linear_checker_services
    found = _invoke(
        runtime,
        "linear.rational_solution.find",
        {"system": _system([[1]], [1])},
        mode=CapabilityMode.EXPLORE,
    )
    monkeypatch.setattr(
        "jacobian.verification.service.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    result = _invoke(
        runtime,
        "linear.rational_solution.verify",
        {"solution_uri": found.output["solution_uri"]},
        mode=CapabilityMode.VERIFY,
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output["status"] == "TIMEOUT"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification_record_uri"] is None
    assert result.assurance.level is not CapabilityAssuranceLevel.VERIFIED
