"""Pinned optional Singular runtime identity for certificate production."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
)
from jacobian.provider_runtime import (
    ProviderRuntimeError,
    ProviderRuntimeErrorCode,
    _platform_tag,
    _sha256_file,
    _unavailable_runtime,
)
from jacobian.worker_environment import worker_environment

SINGULAR_VERSION = "4.4.1p5"
_VERSION_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])(?P<version>[0-9]+\.[0-9]+\.[0-9]+p[0-9]+(?:[-+][0-9A-Za-z.]+)?)(?![0-9A-Za-z])"
)


def singular_provider_runtime(
    executable: str | Path = "Singular",
) -> ProviderObservation:
    """Inspect the exact pinned Singular command-line producer."""

    resolved_name = shutil.which(os.fspath(executable))
    if resolved_name is None:
        return _unavailable_runtime(
            provider="singular",
            install_tier=ProviderInstallTier.T2,
            license_id="GPL-2.0-or-later",
            diagnostic=f"The pinned Singular {SINGULAR_VERSION} executable is unavailable.",
        )
    try:
        resolved = Path(resolved_name).resolve(strict=True)
        completed = execute_process(
            ProcessRequest(
                executable=str(resolved),
                arguments=("--version",),
                environment=worker_environment(locale="C"),
                cwd=str(Path.cwd()),
                timeout_seconds=5,
                stdin_bytes=b"",
                stdout_limit_bytes=16_384,
                stderr_limit_bytes=16_384,
            )
        )
        version_text = (completed.stdout + completed.stderr).decode("utf-8")
        version_match = _VERSION_PATTERN.search(version_text)
        if (
            completed.termination is not ProcessTermination.EXITED
            or completed.returncode != 0
            or version_match is None
            or version_match.group("version") != SINGULAR_VERSION
        ):
            raise ProviderRuntimeError(
                "Singular version probe did not match the pin",
                code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
            )
        digest = _sha256_file(resolved)
    except (OSError, UnicodeDecodeError, ProviderRuntimeError):
        return _unavailable_runtime(
            provider="singular",
            install_tier=ProviderInstallTier.T2,
            license_id="GPL-2.0-or-later",
            diagnostic=f"The pinned Singular {SINGULAR_VERSION} executable is unavailable.",
        )
    return ProviderObservation(
        provider="singular",
        availability=ProviderAvailability.AVAILABLE,
        version=SINGULAR_VERSION,
        digest=digest,
        digest_kind=ProviderDigestKind.EXECUTABLE,
        platform=_platform_tag(),
        install_tier=ProviderInstallTier.T2,
        license_id="GPL-2.0-or-later",
        features=("lift", "groebner-basis", "nullstellensatz-certificate"),
        configuration={
            "executable": str(resolved),
            "profile": "jacobian.nullstellensatz.chart-cover/v1",
        },
    )


__all__ = ["SINGULAR_VERSION", "singular_provider_runtime"]
