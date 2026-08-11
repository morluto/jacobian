from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.domains.polynomial_nullstellensatz import (
    build_nullstellensatz_core_bundle,
)
from jacobian.domains.polynomial_nullstellensatz.bundle import CORE_DOMAIN_ID
from jacobian.domains.polynomial_nullstellensatz.core import MATERIALIZE_CAPABILITY_ID
from jacobian.domains.polynomial_nullstellensatz.singular import (
    PRODUCE_CAPABILITY_ID,
    install_singular_producer,
)
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.process_policy import ProcessResult, ProcessTermination
from jacobian.providers.singular_runtime import singular_provider_runtime


def _runtime() -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider="singular",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="4.4.1p5",
        digest="sha256:" + "8" * 64,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform="test-platform",
        install_tier=CapabilityInstallTier.T2,
        license_id="GPL-2.0-or-later",
        features=("nullstellensatz-certificate",),
        configuration={"executable": "/usr/bin/false"},
    )


def _install(services: DomainTestServices) -> None:
    result = DomainBundleInstaller(services.installation).install(
        PortfolioPlan(domain_bundles=(build_nullstellensatz_core_bundle(),))
    )
    installed = install_singular_producer(
        services.installation,
        result.installed[CORE_DOMAIN_ID],
        _runtime(),
    )
    for adapter in installed.adapters:
        services.installation.register_capability(adapter)


def _invoke(
    services: DomainTestServices,
    capability_id: str,
    payload: dict[str, Any],
) -> CapabilityResult:
    return services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            input=payload,
        )
    )


@pytest.mark.parametrize(
    ("completed", "expected_status", "expected_code"),
    (
        (
            ProcessResult(
                termination=ProcessTermination.TIMED_OUT,
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
            ),
            "TIMEOUT",
            "SINGULAR_TIMEOUT",
        ),
        (
            ProcessResult(
                termination=ProcessTermination.CANCELLED,
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
            ),
            "CANCELLED",
            "SINGULAR_CANCELLED",
        ),
        (
            ProcessResult(
                termination=ProcessTermination.EXITED,
                returncode=0,
                stdout=b"not the bounded protocol",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
            ),
            "ERROR",
            "SINGULAR_PROTOCOL_INVALID",
        ),
        (
            ProcessResult(
                termination=ProcessTermination.OUTPUT_LIMIT_EXCEEDED,
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=True,
                stderr_exceeded=False,
            ),
            "ERROR",
            "SINGULAR_OUTPUT_LIMIT_EXCEEDED",
        ),
    ),
)
def test_singular_boundary_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed: ProcessResult,
    expected_status: str,
    expected_code: str,
) -> None:
    monkeypatch.setattr(
        "jacobian.domains.polynomial_nullstellensatz.singular.execute_process",
        lambda *_args, **_kwargs: completed,
    )
    with open_domain_services(tmp_path) as services:
        _install(services)
        materialized = _invoke(
            services,
            MATERIALIZE_CAPABILITY_ID,
            {},
        )

        result = _invoke(
            services,
            PRODUCE_CAPABILITY_ID,
            {"system_uri": materialized.output["system_uri"]},
        )

        assert result.execution.status.value == expected_status
        assert result.output["error"]["code"] == expected_code
        assert result.assurance.level.value == "HEURISTIC"
        assert not result.artifact_uris


def test_missing_singular_only_marks_provider_unavailable() -> None:
    runtime = singular_provider_runtime("definitely-missing-jacobian-singular")

    assert runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    assert "4.4.1p5" in (runtime.diagnostic or "")


@pytest.mark.parametrize(
    ("version_text", "available"),
    (
        ("Singular 4.4.1p5\n", True),
        ("Singular 4.4.1p50\n", False),
        ("Singular 4.4.1p5-custom\n", False),
    ),
)
def test_singular_version_probe_matches_the_pinned_version_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version_text: str,
    available: bool,
) -> None:
    executable = tmp_path / "Singular"
    executable.write_bytes(b"test executable")
    monkeypatch.setattr(
        "jacobian.providers.singular_runtime.shutil.which",
        lambda _name: str(executable),
    )
    monkeypatch.setattr(
        "jacobian.providers.singular_runtime.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.EXITED,
            returncode=0,
            stdout=version_text.encode(),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    runtime = singular_provider_runtime()

    assert runtime.availability is (
        CapabilityProviderAvailability.AVAILABLE
        if available
        else CapabilityProviderAvailability.UNAVAILABLE
    )
