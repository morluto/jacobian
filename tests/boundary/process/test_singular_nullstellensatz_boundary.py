from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.contracts.operations import (
    OperationRequest,
    OperationResult,
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.domains.polynomial_nullstellensatz.core import (
    MATERIALIZE_OPERATION_ID,
    install_nullstellensatz_core,
)
from jacobian.domains.polynomial_nullstellensatz.singular import (
    PRODUCE_OPERATION_ID,
    bind_selected_singular_producer,
    install_singular_producer,
)
from jacobian.operation_registry import supports_selected_operation
from jacobian.process_policy import ProcessResult, ProcessTermination
from jacobian.provider_runtime import known_provider_runtime
from jacobian.providers.singular_runtime import singular_provider_runtime


def _runtime() -> ProviderObservation:
    return ProviderObservation(
        provider="singular",
        availability=ProviderAvailability.AVAILABLE,
        version="4.4.1p5",
        digest="sha256:" + "8" * 64,
        digest_kind=ProviderDigestKind.EXECUTABLE,
        platform="test-platform",
        install_tier=ProviderInstallTier.T2,
        license_id="GPL-2.0-or-later",
        features=("nullstellensatz-certificate",),
        configuration={"executable": "/usr/bin/false"},
    )


def _install(services: DomainTestServices) -> None:
    core_runtime = known_provider_runtime(
        "jacobian.nullstellensatz-core",
        features=(
            "normalized-jacobian-degree-slice",
            "rabinowitsch-chart-cover",
            "independent-exact-replay",
        ),
    )
    with atomic_installation(services.core):
        core = install_nullstellensatz_core(services.installation, core_runtime)
        for adapter in core.adapters:
            services.installation.register_operation(adapter)
        singular = install_singular_producer(services.installation, core, _runtime())
        for adapter in singular.adapters:
            services.installation.register_operation(adapter)


def _invoke(
    services: DomainTestServices,
    operation_id: str,
    payload: dict[str, Any],
) -> OperationResult:
    return services.core.operations.invoke(
        OperationRequest(
            operation_id=operation_id,
            input=payload,
        )
    )


def test_singular_producer_has_a_lazy_selected_binding(tmp_path: Path) -> None:
    with open_domain_services(tmp_path) as services:
        adapter = bind_selected_singular_producer(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            _runtime(),
        )

        assert supports_selected_operation(PRODUCE_OPERATION_ID)
        assert adapter.descriptor.operation_id == PRODUCE_OPERATION_ID


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
            MATERIALIZE_OPERATION_ID,
            {},
        )

        result = _invoke(
            services,
            PRODUCE_OPERATION_ID,
            {"system_uri": materialized.output["system_uri"]},
        )

        assert result.execution.status.value == expected_status
        assert result.output["error"]["code"] == expected_code
        assert not result.artifact_uris


def test_missing_singular_only_marks_provider_unavailable() -> None:
    runtime = singular_provider_runtime("definitely-missing-jacobian-singular")

    assert runtime.availability is ProviderAvailability.UNAVAILABLE
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
        ProviderAvailability.AVAILABLE
        if available
        else ProviderAvailability.UNAVAILABLE
    )
