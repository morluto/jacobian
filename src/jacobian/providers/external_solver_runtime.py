"""Provider-owned probes for optional SAT and SMT runtimes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from jacobian.canonical import loads_strict_json
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
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
    python_distribution_provider_runtime,
)
from jacobian.worker_environment import worker_environment

CADICAL_VERSION = "3.0.1"
CARCARA_SOURCE_COMMIT = "394edbb15ba95c47893f1d821fddde7e016af178"
CARCARA_SOURCE_REPOSITORY = "https://github.com/ufmg-smite/carcara"
CARCARA_VERSION = "1.1.0"
CVC5_VERSION = "1.3.4"
DRAT_TRIM_RELEASE_TAG = "v05.22.2023"
DRAT_TRIM_SOURCE_COMMIT = "2e5e29cb0019d5cfd547d4208dca1b3ec290349f"
DRAT_TRIM_SOURCE_REPOSITORY = "https://github.com/marijnheule/drat-trim"


def cadical_provider_runtime(
    executable: str | Path = "cadical",
) -> CapabilityProviderRuntime:
    """Inspect the exact pinned CaDiCaL competition CLI runtime."""

    resolved_name = shutil.which(os.fspath(executable))
    if resolved_name is None:
        return _unavailable_runtime(
            provider="cadical",
            install_tier=CapabilityInstallTier.T2,
            license_id="MIT",
            diagnostic=(
                f"The pinned CaDiCaL {CADICAL_VERSION} executable is unavailable."
            ),
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
                stdout_limit_bytes=1024,
                stderr_limit_bytes=4096,
            )
        )
        if completed.termination is not ProcessTermination.EXITED:
            raise ProviderRuntimeError(
                "CaDiCaL version probe did not match the pin",
                code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
            )
        version = completed.stdout.decode("ascii").strip()
        if completed.returncode != 0 or version != CADICAL_VERSION:
            raise ProviderRuntimeError(
                "CaDiCaL version probe did not match the pin",
                code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
            )
        digest = _sha256_file(resolved)
    except (OSError, UnicodeDecodeError, ProviderRuntimeError):
        return _unavailable_runtime(
            provider="cadical",
            install_tier=CapabilityInstallTier.T2,
            license_id="MIT",
            diagnostic=(
                f"The pinned CaDiCaL {CADICAL_VERSION} executable is unavailable."
            ),
        )
    return CapabilityProviderRuntime(
        provider="cadical",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version=CADICAL_VERSION,
        digest=digest,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform=_platform_tag(),
        install_tier=CapabilityInstallTier.T2,
        license_id="MIT",
        features=("competition-cli", "total-model", "drat-text-proof"),
        configuration={
            "executable": str(resolved),
            "projection": "jacobian.dimacs.cnf/v1",
            "proof_format": "drat-text/v1",
        },
    )


def cvc5_provider_runtime() -> CapabilityProviderRuntime:
    """Inspect the exact optional cvc5 Python distribution used for Alethe."""

    runtime = python_distribution_provider_runtime(
        "cvc5",
        distribution_name="cvc5",
        import_name="cvc5",
        required_attributes=(
            "InputParser",
            "ProofComponent",
            "ProofFormat",
            "Solver",
        ),
        install_tier=CapabilityInstallTier.T1,
        license_id="BSD-3-Clause",
        features=("smt-lib-2.6", "alethe-proof-production"),
        configuration={
            "profile": "jacobian.smtlib2.qf-unsat/v1",
            "proof_format": "cvc5.alethe/1.3.4",
        },
    )
    if (
        runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.version != CVC5_VERSION
    ):
        return _unavailable_runtime(
            provider="cvc5",
            install_tier=CapabilityInstallTier.T1,
            license_id="BSD-3-Clause",
            diagnostic=(
                f"The pinned cvc5 {CVC5_VERSION} Python distribution is unavailable."
            ),
        )
    return runtime


def drat_trim_provider_runtime(
    executable: str | Path = "drat-trim",
    *,
    provenance_file: str | Path | None = None,
) -> CapabilityProviderRuntime:
    """Inspect an operator-provenanced pinned DRAT-trim runtime."""

    resolved_name = shutil.which(os.fspath(executable))
    if resolved_name is None:
        return _unavailable_runtime(
            provider="drat-trim",
            install_tier=CapabilityInstallTier.T2,
            license_id="MIT",
            diagnostic=(
                f"The pinned DRAT-trim {DRAT_TRIM_RELEASE_TAG} runtime is unavailable."
            ),
        )
    try:
        resolved = Path(resolved_name).resolve(strict=True)
        manifest_path = (
            Path(provenance_file).resolve(strict=True)
            if provenance_file is not None
            else resolved.with_name(resolved.name + ".jacobian-runtime.json").resolve(
                strict=True
            )
        )
        manifest = loads_strict_json(manifest_path.read_bytes())
        expected_manifest = {
            "runtime_manifest_version": "1",
            "provider": "drat-trim",
            "release_tag": DRAT_TRIM_RELEASE_TAG,
            "source_repository": DRAT_TRIM_SOURCE_REPOSITORY,
            "source_commit": DRAT_TRIM_SOURCE_COMMIT,
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {*expected_manifest, "executable_sha256"}
            or any(
                manifest.get(key) != value for key, value in expected_manifest.items()
            )
            or not isinstance(manifest.get("executable_sha256"), str)
        ):
            raise ProviderRuntimeError(
                "DRAT-trim provenance is invalid",
                code=ProviderRuntimeErrorCode.MALFORMED_RUNTIME,
            )
        digest = _sha256_file(resolved)
        if manifest["executable_sha256"] != digest:
            raise ProviderRuntimeError(
                "DRAT-trim executable digest changed",
                code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
            )
        completed = execute_process(
            ProcessRequest(
                executable=str(resolved),
                arguments=("-h",),
                environment=worker_environment(locale="C"),
                cwd=str(Path.cwd()),
                timeout_seconds=5,
                stdin_bytes=b"",
                stdout_limit_bytes=16_000,
                stderr_limit_bytes=16_000,
            )
        )
        if (
            completed.termination is not ProcessTermination.EXITED
            or completed.returncode != 0
            or b"usage: drat-trim" not in completed.stdout
        ):
            raise ProviderRuntimeError(
                "DRAT-trim health probe failed",
                code=ProviderRuntimeErrorCode.READINESS_FAILED,
            )
    except (OSError, ProviderRuntimeError, ValueError):
        return _unavailable_runtime(
            provider="drat-trim",
            install_tier=CapabilityInstallTier.T2,
            license_id="MIT",
            diagnostic=(
                f"The pinned DRAT-trim {DRAT_TRIM_RELEASE_TAG} runtime and "
                "operator provenance are unavailable."
            ),
        )
    return CapabilityProviderRuntime(
        provider="drat-trim",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version=DRAT_TRIM_RELEASE_TAG,
        digest=digest,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform=_platform_tag(),
        install_tier=CapabilityInstallTier.T2,
        license_id="MIT",
        features=("drat-text/v1", "unsat-proof-replay"),
        configuration={
            "executable": str(resolved),
            "provenance_file": str(manifest_path),
            "source_repository": DRAT_TRIM_SOURCE_REPOSITORY,
            "source_commit": DRAT_TRIM_SOURCE_COMMIT,
        },
    )


def carcara_provider_runtime(
    executable: str | Path = "carcara",
    *,
    provenance_file: str | Path | None = None,
) -> CapabilityProviderRuntime:
    """Inspect the exact operator-provenanced Carcara Alethe checker runtime."""

    resolved_name = shutil.which(os.fspath(executable))
    if resolved_name is None:
        return _unavailable_runtime(
            provider="carcara",
            install_tier=CapabilityInstallTier.T2,
            license_id="Apache-2.0",
            diagnostic=(
                f"The pinned Carcara {CARCARA_VERSION} runtime is unavailable."
            ),
        )
    try:
        resolved = Path(resolved_name).resolve(strict=True)
        manifest_path = (
            Path(provenance_file).resolve(strict=True)
            if provenance_file is not None
            else resolved.with_name(resolved.name + ".jacobian-runtime.json").resolve(
                strict=True
            )
        )
        manifest = loads_strict_json(manifest_path.read_bytes())
        expected_manifest = {
            "runtime_manifest_version": "1",
            "provider": "carcara",
            "version": CARCARA_VERSION,
            "source_repository": CARCARA_SOURCE_REPOSITORY,
            "source_commit": CARCARA_SOURCE_COMMIT,
            "compatible_cvc5_version": CVC5_VERSION,
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {*expected_manifest, "executable_sha256"}
            or any(
                manifest.get(key) != value for key, value in expected_manifest.items()
            )
            or not isinstance(manifest.get("executable_sha256"), str)
        ):
            raise ProviderRuntimeError(
                "Carcara provenance is invalid",
                code=ProviderRuntimeErrorCode.MALFORMED_RUNTIME,
            )
        digest = _sha256_file(resolved)
        if manifest["executable_sha256"] != digest:
            raise ProviderRuntimeError(
                "Carcara executable digest changed",
                code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
            )
        environment = worker_environment(locale="C")
        version = execute_process(
            ProcessRequest(
                executable=str(resolved),
                arguments=("--version",),
                environment=environment,
                cwd=str(Path.cwd()),
                timeout_seconds=5,
                stdin_bytes=b"",
                stdout_limit_bytes=16_000,
                stderr_limit_bytes=16_000,
            )
        )
        help_result = execute_process(
            ProcessRequest(
                executable=str(resolved),
                arguments=("check", "--help"),
                environment=environment,
                cwd=str(Path.cwd()),
                timeout_seconds=5,
                stdin_bytes=b"",
                stdout_limit_bytes=64_000,
                stderr_limit_bytes=16_000,
            )
        )
        expected_version = (
            f"carcara {CARCARA_VERSION} [git master {CARCARA_SOURCE_COMMIT[:7]}]\n"
        ).encode("ascii")
        required_help = (
            b"--strict-parsing",
            b"--parse-hole-args",
            b"--allow-int-real-subtyping",
            b"--expand-let-bindings",
        )
        if (
            version.termination is not ProcessTermination.EXITED
            or version.returncode != 0
            or version.stdout != expected_version
            or version.stderr
            or help_result.termination is not ProcessTermination.EXITED
            or help_result.returncode != 0
            or help_result.stderr
            or any(flag not in help_result.stdout for flag in required_help)
        ):
            raise ProviderRuntimeError(
                "Carcara health probe failed",
                code=ProviderRuntimeErrorCode.READINESS_FAILED,
            )
    except (OSError, ProviderRuntimeError, ValueError):
        return _unavailable_runtime(
            provider="carcara",
            install_tier=CapabilityInstallTier.T2,
            license_id="Apache-2.0",
            diagnostic=(
                f"The pinned Carcara {CARCARA_VERSION} runtime and operator "
                "provenance are unavailable."
            ),
        )
    return CapabilityProviderRuntime(
        provider="carcara",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version=CARCARA_VERSION,
        digest=digest,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform=_platform_tag(),
        install_tier=CapabilityInstallTier.T2,
        license_id="Apache-2.0",
        features=(
            "alethe-proof-replay",
            "strict-parsing",
            "reject-holes",
        ),
        configuration={
            "executable": str(resolved),
            "provenance_file": str(manifest_path),
            "source_repository": CARCARA_SOURCE_REPOSITORY,
            "source_commit": CARCARA_SOURCE_COMMIT,
            "compatible_cvc5_version": CVC5_VERSION,
            "problem_profile": "jacobian.smtlib2.qf-unsat/v1",
            "accepted_logic": "QF_UF",
            "proof_format": "cvc5.alethe/1.3.4",
            "command_flags": [
                "--strict-parsing",
                "--parse-hole-args",
                "--allow-int-real-subtyping",
                "--expand-let-bindings",
            ],
        },
    )
