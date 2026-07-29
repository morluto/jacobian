"""Fail-closed runtime identity probes for capability providers."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import shutil
import sysconfig
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from jacobian.bounded_process import run_bounded_process
from jacobian.canonical import canonicalize_json, loads_strict_json
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.contracts.polynomial_expressions import (
    SYMPY_POLYNOMIAL_NORMALIZATION_CONFIGURATION,
)
from jacobian.implementation import ImplementationError, package_source_digest

CADICAL_VERSION = "3.0.1"
CARCARA_SOURCE_COMMIT = "394edbb15ba95c47893f1d821fddde7e016af178"
CARCARA_SOURCE_REPOSITORY = "https://github.com/ufmg-smite/carcara"
CARCARA_VERSION = "1.1.0"
CVC5_VERSION = "1.3.4"
DRAT_TRIM_RELEASE_TAG = "v05.22.2023"
DRAT_TRIM_SOURCE_COMMIT = "2e5e29cb0019d5cfd547d4208dca1b3ec290349f"
DRAT_TRIM_SOURCE_REPOSITORY = "https://github.com/marijnheule/drat-trim"
PYTHON_FLINT_VERSION = "0.9.0"
NETWORKX_VERSION = "3.6.1"
PYTHON_FLINT_HNF_FLINT_VERSION = "3.6.0"
PYTHON_FLINT_LLL_CONFIGURATION = {
    "domain": "ZZ",
    "operation": "fmpz_mat.lll(transform=True)",
    "flint_library_version": PYTHON_FLINT_HNF_FLINT_VERSION,
    "maximum_rows": 32,
    "maximum_columns": 32,
    "maximum_decimal_digits_per_entry": 256,
    "representation": "zbasis",
    "gram": "exact",
    "delta_double": "0.99",
    "eta_double": "0.51",
    "relation": "L=T*A",
}
SYMPY_VERSION = "1.14.0"
SYMPY_POLYNOMIAL_WORKER_PROTOCOL = "jacobian.sympy-polynomial-normalization/v1"
Z3_SOLVER_VERSION = "5.0.0.0"


class ProviderRuntimeError(RuntimeError):
    """Raised when a required provider identity cannot be inspected safely."""


@cache
def _platform_tag() -> str:
    return sysconfig.get_platform()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _distribution_record_digest(
    distribution: importlib.metadata.Distribution,
) -> str:
    rows: list[str] = []
    hashed = 0
    for package_path in sorted(distribution.files or (), key=str):
        file_hash = package_path.hash
        if file_hash is None:
            hash_value = "-"
        else:
            hash_value = f"{file_hash.mode}:{file_hash.value}"
            hashed += 1
        size = "-" if package_path.size is None else str(package_path.size)
        rows.append(f"{package_path}\0{hash_value}\0{size}\n")
    if not rows or not hashed:
        raise ProviderRuntimeError(
            "the installed Python distribution has no hashed RECORD manifest"
        )
    digest = hashlib.sha256("".join(rows).encode()).hexdigest()
    return f"sha256:{digest}"


def _license_files(
    distribution: importlib.metadata.Distribution,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(package_path).replace("\\", "/")
            for package_path in distribution.files or ()
            if "license" in package_path.name.lower()
        )
    )


@cache
def _jacobian_identity() -> tuple[str, str, tuple[str, ...]]:
    try:
        distribution = importlib.metadata.distribution("jacobian")
        digest = package_source_digest("jacobian.capabilities:CapabilityService")
    except (importlib.metadata.PackageNotFoundError, ImplementationError) as exc:
        raise ProviderRuntimeError(
            "the Jacobian source runtime could not be identified"
        ) from exc
    return distribution.version, digest, _license_files(distribution)


def _inspect_python_distribution_identity(
    distribution_name: str,
    import_name: str,
    required_attributes: tuple[str, ...],
) -> tuple[str, str, tuple[str, ...]]:
    """Read immutable distribution metadata without importing its implementation.

    ``import_name`` and ``required_attributes`` remain bound into the returned
    runtime contract. The provider loader validates them on first use; identity
    measurement itself must stay import-free so runtime assembly does not load
    every optional mathematical backend.
    """

    del required_attributes
    try:
        if importlib.util.find_spec(import_name) is None:
            raise ProviderRuntimeError(
                f"the {distribution_name} import target is unavailable"
            )
        distribution = importlib.metadata.distribution(distribution_name)
        digest = _distribution_record_digest(distribution)
    except (
        ImportError,
        importlib.metadata.PackageNotFoundError,
        ProviderRuntimeError,
    ) as exc:
        raise ProviderRuntimeError(
            f"the {distribution_name} distribution identity is unavailable"
        ) from exc
    return distribution.version, digest, _license_files(distribution)


@cache
def _python_distribution_identity(
    distribution_name: str,
    import_name: str,
    required_attributes: tuple[str, ...],
) -> tuple[str, str, tuple[str, ...]]:
    return _inspect_python_distribution_identity(
        distribution_name,
        import_name,
        required_attributes,
    )


def _unavailable_runtime(
    *,
    provider: str,
    install_tier: CapabilityInstallTier,
    license_id: str,
    diagnostic: str,
) -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider=provider,
        availability=CapabilityProviderAvailability.UNAVAILABLE,
        platform=_platform_tag(),
        install_tier=install_tier,
        license_id=license_id,
        diagnostic=diagnostic,
    )


def composite_provider_runtime(
    provider: str,
    *,
    components: tuple[CapabilityProviderRuntime, ...],
    features: tuple[str, ...] = (),
    checker_ids: tuple[str, ...] = (),
    configuration: Mapping[str, Any] | None = None,
) -> CapabilityProviderRuntime:
    """Bind one capability provider to every runtime component it executes."""

    if not components:
        raise ValueError("a composite provider requires at least one component")
    component_providers = tuple(component.provider for component in components)
    if len(set(component_providers)) != len(component_providers):
        raise ValueError("composite provider components must have unique identities")
    install_tiers = tuple(CapabilityInstallTier)
    install_tier = max(
        (component.install_tier for component in components),
        key=install_tiers.index,
    )
    unavailable = tuple(
        component
        for component in components
        if component.availability is CapabilityProviderAvailability.UNAVAILABLE
    )
    if unavailable:
        names = ", ".join(component.provider for component in unavailable)
        return _unavailable_runtime(
            provider=provider,
            install_tier=install_tier,
            license_id=" AND ".join(
                sorted({component.license_id for component in components})
            ),
            diagnostic=f"Required composite provider components are unavailable: {names}.",
        )

    component_records = [component.model_dump(mode="json") for component in components]
    digest = hashlib.sha256(canonicalize_json(component_records)).hexdigest()
    return CapabilityProviderRuntime(
        provider=provider,
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="composite-1",
        digest=f"sha256:{digest}",
        digest_kind=CapabilityProviderDigestKind.COMPOSITE,
        platform=_platform_tag(),
        install_tier=install_tier,
        license_id=" AND ".join(
            sorted({component.license_id for component in components})
        ),
        features=features,
        checker_ids=checker_ids,
        configuration={
            "components": component_records,
            **dict(configuration or {}),
        },
    )


def jacobian_provider_runtime(
    provider: str,
    *,
    features: tuple[str, ...] = (),
    checker_ids: tuple[str, ...] = (),
    configuration: Mapping[str, Any] | None = None,
) -> CapabilityProviderRuntime:
    """Identify a source-backed provider implemented inside Jacobian."""

    try:
        version, digest, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        return _unavailable_runtime(
            provider=provider,
            install_tier=CapabilityInstallTier.T0,
            license_id="MIT",
            diagnostic="The Jacobian source runtime could not be identified.",
        )
    return CapabilityProviderRuntime(
        provider=provider,
        availability=CapabilityProviderAvailability.AVAILABLE,
        version=version,
        digest=digest,
        digest_kind=CapabilityProviderDigestKind.SOURCE_TREE,
        platform=_platform_tag(),
        install_tier=CapabilityInstallTier.T0,
        license_id="MIT",
        license_files=license_files,
        features=features,
        checker_ids=checker_ids,
        configuration=dict(configuration or {}),
    )


def source_provider_runtime(
    provider: str,
    *,
    version: str,
    entrypoint: str,
    install_tier: CapabilityInstallTier,
    license_id: str,
    license_files: tuple[str, ...] = (),
    features: tuple[str, ...] = (),
    checker_ids: tuple[str, ...] = (),
    configuration: Mapping[str, Any] | None = None,
) -> CapabilityProviderRuntime:
    """Identify an operator-installed source package without importing its code."""

    try:
        digest = package_source_digest(entrypoint)
    except ImplementationError:
        return _unavailable_runtime(
            provider=provider,
            install_tier=install_tier,
            license_id=license_id,
            diagnostic="The provider source package could not be identified.",
        )
    return CapabilityProviderRuntime(
        provider=provider,
        availability=CapabilityProviderAvailability.AVAILABLE,
        version=version,
        digest=digest,
        digest_kind=CapabilityProviderDigestKind.SOURCE_TREE,
        platform=_platform_tag(),
        install_tier=install_tier,
        license_id=license_id,
        license_files=license_files,
        features=features,
        checker_ids=checker_ids,
        configuration={
            "entrypoint": entrypoint,
            **dict(configuration or {}),
        },
    )


def python_distribution_provider_runtime(
    provider: str,
    *,
    distribution_name: str,
    import_name: str,
    required_attributes: tuple[str, ...],
    install_tier: CapabilityInstallTier,
    license_id: str,
    features: tuple[str, ...] = (),
    checker_ids: tuple[str, ...] = (),
    configuration: Mapping[str, Any] | None = None,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify one installed Python distribution without trusting an import alone."""

    try:
        identity = (
            _inspect_python_distribution_identity
            if refresh
            else _python_distribution_identity
        )
        version, digest, license_files = identity(
            distribution_name, import_name, required_attributes
        )
    except ProviderRuntimeError:
        return _unavailable_runtime(
            provider=provider,
            install_tier=install_tier,
            license_id=license_id,
            diagnostic=(
                f"The {distribution_name} provider is not installed and healthy."
            ),
        )
    return CapabilityProviderRuntime(
        provider=provider,
        availability=CapabilityProviderAvailability.AVAILABLE,
        version=version,
        digest=digest,
        digest_kind=CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        platform=_platform_tag(),
        install_tier=install_tier,
        license_id=license_id,
        license_files=license_files,
        features=features,
        checker_ids=checker_ids,
        configuration={
            **dict(configuration or {}),
            "distribution": distribution_name,
        },
        distribution_import_name=import_name,
        distribution_required_attributes=required_attributes,
    )


def require_provider_runtime_unchanged(
    runtime: CapabilityProviderRuntime,
) -> None:
    """Remeasure every immutable component of an authorized provider runtime."""

    if (
        runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.digest is None
        or runtime.digest_kind is None
    ):
        raise ProviderRuntimeError("provider runtime identity is incomplete")

    if runtime.digest_kind is CapabilityProviderDigestKind.EXECUTABLE:
        executable = runtime.configuration.get("executable")
        if (
            not isinstance(executable, str)
            or _sha256_file(Path(executable)) != runtime.digest
        ):
            raise ProviderRuntimeError("provider executable identity changed")
        return

    if runtime.digest_kind is CapabilityProviderDigestKind.SOURCE_TREE:
        entrypoint = runtime.configuration.get("entrypoint")
        if not isinstance(entrypoint, str):
            raise ProviderRuntimeError("provider source identity is incomplete")
        try:
            measured_source = package_source_digest(entrypoint)
        except ImplementationError as exc:
            raise ProviderRuntimeError("provider source is unavailable") from exc
        if measured_source != runtime.digest:
            raise ProviderRuntimeError("provider source identity changed")
        return

    if runtime.digest_kind is CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD:
        distribution = runtime.configuration.get("distribution")
        import_name = runtime.distribution_import_name
        if not isinstance(distribution, str) or import_name is None:
            raise ProviderRuntimeError("Python distribution identity is incomplete")
        version, digest, _license_files = _inspect_python_distribution_identity(
            distribution,
            import_name,
            runtime.distribution_required_attributes,
        )
        if version != runtime.version or digest != runtime.digest:
            raise ProviderRuntimeError("Python distribution identity changed")
        return

    components = runtime.configuration.get("components")
    if not isinstance(components, list) or not components:
        raise ProviderRuntimeError("composite provider identity is incomplete")
    component_runtimes = tuple(
        CapabilityProviderRuntime.model_validate(component) for component in components
    )
    for component in component_runtimes:
        require_provider_runtime_unchanged(component)
    measured = composite_provider_runtime(
        runtime.provider,
        components=component_runtimes,
        features=runtime.features,
        checker_ids=runtime.checker_ids,
        configuration={
            key: value
            for key, value in runtime.configuration.items()
            if key != "components"
        },
    )
    if measured.digest != runtime.digest:
        raise ProviderRuntimeError("composite provider identity changed")


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
        completed = run_bounded_process(
            [str(resolved), "--version"],
            input_bytes=b"",
            timeout_seconds=5,
            environment={
                **os.environ,
                "LANG": "C",
                "LC_ALL": "C",
                "TZ": "UTC",
            },
            stdout_limit=1024,
            stderr_limit=4096,
        )
        version = completed.stdout.decode("ascii").strip()
        if (
            completed.timed_out
            or completed.stdout_exceeded
            or completed.stderr_exceeded
            or completed.returncode != 0
            or version != CADICAL_VERSION
        ):
            raise ProviderRuntimeError("CaDiCaL version probe did not match the pin")
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
            raise ProviderRuntimeError("DRAT-trim provenance is invalid")
        digest = _sha256_file(resolved)
        if manifest["executable_sha256"] != digest:
            raise ProviderRuntimeError("DRAT-trim executable digest changed")
        completed = run_bounded_process(
            [str(resolved), "-h"],
            input_bytes=b"",
            timeout_seconds=5,
            environment={
                **os.environ,
                "LANG": "C",
                "LC_ALL": "C",
                "TZ": "UTC",
            },
            stdout_limit=16_000,
            stderr_limit=16_000,
        )
        if (
            completed.timed_out
            or completed.stdout_exceeded
            or completed.stderr_exceeded
            or completed.returncode != 0
            or b"usage: drat-trim" not in completed.stdout
        ):
            raise ProviderRuntimeError("DRAT-trim health probe failed")
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
            raise ProviderRuntimeError("Carcara provenance is invalid")
        digest = _sha256_file(resolved)
        if manifest["executable_sha256"] != digest:
            raise ProviderRuntimeError("Carcara executable digest changed")
        environment = {
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
        }
        version = run_bounded_process(
            [str(resolved), "--version"],
            input_bytes=b"",
            timeout_seconds=5,
            environment=environment,
            stdout_limit=16_000,
            stderr_limit=16_000,
        )
        help_result = run_bounded_process(
            [str(resolved), "check", "--help"],
            input_bytes=b"",
            timeout_seconds=5,
            environment=environment,
            stdout_limit=64_000,
            stderr_limit=16_000,
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
            version.timed_out
            or version.stdout_exceeded
            or version.stderr_exceeded
            or version.returncode != 0
            or version.stdout != expected_version
            or version.stderr
            or help_result.timed_out
            or help_result.stdout_exceeded
            or help_result.stderr_exceeded
            or help_result.returncode != 0
            or help_result.stderr
            or any(flag not in help_result.stdout for flag in required_help)
        ):
            raise ProviderRuntimeError("Carcara health probe failed")
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


def known_provider_runtime(
    provider: str,
    *,
    features: tuple[str, ...] = (),
    checker_ids: tuple[str, ...] = (),
    configuration: Mapping[str, Any] | None = None,
) -> CapabilityProviderRuntime:
    """Resolve runtime identity for a built-in provider family."""

    if provider == "jacobian.networkx":
        runtime = python_distribution_provider_runtime(
            provider,
            distribution_name="networkx",
            import_name="networkx",
            required_attributes=("Graph", "graph_atlas_g"),
            install_tier=CapabilityInstallTier.T0,
            license_id="BSD-3-Clause",
            features=features,
            checker_ids=checker_ids,
            configuration=configuration,
        )
        if (
            runtime.availability is CapabilityProviderAvailability.AVAILABLE
            and runtime.version != NETWORKX_VERSION
        ):
            return _unavailable_runtime(
                provider=provider,
                install_tier=CapabilityInstallTier.T0,
                license_id="BSD-3-Clause",
                diagnostic=(
                    "NetworkX is installed but does not match the pinned "
                    f"{NETWORKX_VERSION} graph-operation profile."
                ),
            )
        return runtime
    if provider == "jacobian.sympy":
        runtime = python_distribution_provider_runtime(
            provider,
            distribution_name="sympy",
            import_name="sympy",
            required_attributes=("Matrix", "Poly"),
            install_tier=CapabilityInstallTier.T0,
            license_id="BSD-3-Clause",
            features=features,
            checker_ids=checker_ids,
            configuration=configuration,
        )
        if (
            runtime.availability is CapabilityProviderAvailability.AVAILABLE
            and runtime.version != SYMPY_VERSION
        ):
            return _unavailable_runtime(
                provider=provider,
                install_tier=CapabilityInstallTier.T0,
                license_id="BSD-3-Clause",
                diagnostic=(
                    "SymPy is installed but does not match the pinned "
                    f"{SYMPY_VERSION} exact-operation profile."
                ),
            )
        return runtime
    if provider == "jacobian.z3":
        runtime = python_distribution_provider_runtime(
            provider,
            distribution_name="z3-solver",
            import_name="z3",
            required_attributes=("Real", "Solver"),
            install_tier=CapabilityInstallTier.T0,
            license_id="MIT",
            features=features,
            checker_ids=checker_ids,
            configuration=configuration,
        )
        if (
            runtime.availability is CapabilityProviderAvailability.AVAILABLE
            and runtime.version != Z3_SOLVER_VERSION
        ):
            return _unavailable_runtime(
                provider=provider,
                install_tier=CapabilityInstallTier.T0,
                license_id="MIT",
                diagnostic=(
                    "Z3 is installed but does not match the pinned "
                    f"{Z3_SOLVER_VERSION} graph-search profile."
                ),
            )
        return runtime
    return jacobian_provider_runtime(
        provider,
        features=features,
        checker_ids=checker_ids,
        configuration=configuration,
    )


def sympy_polynomial_normalization_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned typed polynomial-normalization profile."""

    runtime = python_distribution_provider_runtime(
        "jacobian.sympy",
        distribution_name="sympy",
        import_name="sympy",
        required_attributes=("Add", "Mul", "Poly", "Pow", "QQ", "Rational", "Symbol"),
        install_tier=CapabilityInstallTier.T0,
        license_id="BSD-3-Clause",
        features=(
            "typed-polynomial-expression",
            "exact-rational",
            "canonical-sparse-normalization",
        ),
        configuration=SYMPY_POLYNOMIAL_NORMALIZATION_CONFIGURATION,
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != SYMPY_VERSION
    ):
        return _unavailable_runtime(
            provider="jacobian.sympy",
            install_tier=CapabilityInstallTier.T0,
            license_id="BSD-3-Clause",
            diagnostic=(
                "SymPy is installed but does not match the pinned "
                f"{SYMPY_VERSION} polynomial-normalization profile."
            ),
        )
    return runtime


def python_flint_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the exact optional Python-FLINT compatibility profile."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("fmpq", "fmpq_mat"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=(
            "exact-rational",
            "dense-matrix",
            "reduced-row-echelon-form",
        ),
        configuration={
            "domain": "QQ",
            "operation": "fmpq_mat.rref",
            "maximum_rows": 32,
            "maximum_columns": 32,
            "free_variable_policy": "ZERO",
        },
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} compatibility profile."
            ),
        )
    return runtime


def python_flint_exact_checker_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned Python-FLINT API used by exact-domain replay."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=(
            "fmpq",
            "fmpq_mat",
            "fmpq_poly",
            "fmpz",
            "fmpz_mat",
            "fmpz_poly",
        ),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=("exact-domain-independent-replay",),
        configuration={
            "import_name": "flint",
            "flint_library_version": PYTHON_FLINT_HNF_FLINT_VERSION,
        },
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} exact-checker profile."
            ),
        )
    if refresh and runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        try:
            flint = importlib.import_module("flint")
        except (ImportError, OSError):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic=(
                    "The pinned Python-FLINT exact-checker runtime cannot be imported."
                ),
            )
        if getattr(flint, "__FLINT_VERSION__", None) != (
            PYTHON_FLINT_HNF_FLINT_VERSION
        ):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic=(
                    "Python-FLINT is installed but its linked FLINT library does "
                    "not match the pinned exact-checker profile."
                ),
            )
    return runtime


def python_flint_analysis_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned Arb API used by validated real analysis."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("arb", "ctx", "fmpq"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=("arb-ball-arithmetic",),
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} real-analysis profile."
            ),
        )
    return runtime


def python_flint_probability_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned exact-rational API used by finite probability."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("fmpq",),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=(
            "exact-rational-moments",
            "finite-event-probability",
            "finite-conditioning",
            "finite-pushforward",
            "finite-convolution",
        ),
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} finite-probability profile."
            ),
        )
    return runtime


def exact_domain_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Bind independent checker source and its pinned FLINT replay backend."""

    try:
        version, _, _ = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.exact-domain-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The exact-domain checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.exact-domain-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.exact_domain_operations:check_polynomial_gcd"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            features=("clean-process-replay",),
        )
    return composite_provider_runtime(
        "jacobian.exact-domain-checkers",
        components=(
            source,
            python_flint_exact_checker_provider_runtime(refresh=refresh),
        ),
        features=("clean-process-replay", "python-flint"),
        checker_ids=checker_ids,
    )


def graph_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent finite-graph checker source without FLINT."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.graph-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The finite-graph checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.graph-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.graph_exact_operations:"
                "check_graph_induced_tree_maximum"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.graph-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "finite-subset-exhaustive-replay",
            "hamiltonian-path-exhaustive-replay",
            "tutte-berge-barrier-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def probability_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent finite-probability checker source without FLINT."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.probability-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic=(
                "The finite-probability checker source could not be identified."
            ),
        )
    else:
        source = source_provider_runtime(
            "jacobian.probability-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.exact_probability_operations:check_finite_raw_moment"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.probability-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "finite-rational-probability-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def combinatorics_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind recurrence and rational-series checker source without SymPy."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.combinatorics-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The exact combinatorics checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.combinatorics-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.recurrence_series:check_linear_recurrence_evaluation"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.combinatorics-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "linear-recurrence-replay",
            "rational-series-residual-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def topology_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent finite-simplicial-topology checker source."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.topology-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The simplicial-topology checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.topology-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.simplicial_topology:"
                "check_simplicial_complex_materialization"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.topology-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "finite-face-closure-replay",
            "oriented-boundary-replay",
            "prime-field-quotient-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def poset_exact_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the independent finite-poset checker source without NetworkX."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.poset-exact-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The finite-poset checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.poset-exact-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.finite_posets:check_finite_poset_materialization"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=("clean-process-replay", "standard-library-only"),
        )
    return composite_provider_runtime(
        "jacobian.poset-exact-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "finite-poset-closure-replay",
            "dilworth-dual-certificate-replay",
            "complete-ideal-dp-replay",
            "mobius-convolution-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def graded_syzygy_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the standard-library graded-syzygy checker source."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.graded-syzygy-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic="The graded-syzygy checker source could not be identified.",
        )
    else:
        source = source_provider_runtime(
            "jacobian.graded-syzygy-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.jacobian_syzygy:check_graded_jacobian_syzygy"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=(
                "clean-process-replay",
                "exact-rational",
                "standard-library-only",
            ),
        )
    return composite_provider_runtime(
        "jacobian.graded-syzygy-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "graded-coefficient-map-reconstruction",
            "exact-rational-rank-replay",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def projective_arrangement_checker_provider_runtime(
    *,
    checker_ids: tuple[str, ...] = (),
) -> CapabilityProviderRuntime:
    """Bind the standard-library projective-arrangement checker source."""

    try:
        version, _, license_files = _jacobian_identity()
    except ProviderRuntimeError:
        source = _unavailable_runtime(
            provider="jacobian.projective-arrangement-checker-source",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            diagnostic=(
                "The projective-arrangement checker source could not be identified."
            ),
        )
    else:
        source = source_provider_runtime(
            "jacobian.projective-arrangement-checker-source",
            version=version,
            entrypoint=(
                "jacobian_checkers.projective_arrangements:"
                "check_projective_line_arrangement_flats"
            ),
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT",
            license_files=license_files,
            features=(
                "clean-process-replay",
                "exact-rational",
                "standard-library-only",
            ),
        )
    return composite_provider_runtime(
        "jacobian.projective-arrangement-checkers",
        components=(source,),
        features=(
            "clean-process-replay",
            "projective-pair-incidence-exhaustive-replay",
            "exact-rational",
            "standard-library-only",
        ),
        checker_ids=checker_ids,
    )


def python_flint_hnf_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned Python-FLINT integer row-HNF profile."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("fmpz", "fmpz_mat"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=(
            "exact-integer",
            "dense-matrix",
            "row-hermite-normal-form",
            "left-transformation",
        ),
        configuration={
            "domain": "ZZ",
            "operation": "fmpz_mat.hnf(transform=True)",
            "flint_library_version": PYTHON_FLINT_HNF_FLINT_VERSION,
            "maximum_rows": 32,
            "maximum_columns": 32,
            "normal_form_convention": "FLINT_ROW_HNF",
            "relation": "H=U*A",
        },
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} HNF compatibility profile."
            ),
        )
    if refresh and runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        try:
            flint = importlib.import_module("flint")
        except (ImportError, OSError):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic="The pinned Python-FLINT HNF runtime cannot be imported.",
            )
        if getattr(flint, "__FLINT_VERSION__", None) != PYTHON_FLINT_HNF_FLINT_VERSION:
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic=(
                    "Python-FLINT is installed but its linked FLINT library does "
                    "not match the pinned "
                    f"{PYTHON_FLINT_HNF_FLINT_VERSION} HNF profile."
                ),
            )
    return runtime


def python_flint_lll_provider_runtime(
    *,
    refresh: bool = False,
) -> CapabilityProviderRuntime:
    """Identify the pinned Python-FLINT exact-gram LLL profile."""

    runtime = python_distribution_provider_runtime(
        "python-flint",
        distribution_name="python-flint",
        import_name="flint",
        required_attributes=("fmpz", "fmpz_mat"),
        install_tier=CapabilityInstallTier.T1,
        license_id="MIT AND LGPL-3.0-or-later",
        features=(
            "exact-integer",
            "dense-matrix",
            "lll-reduction",
            "left-transformation",
        ),
        configuration=PYTHON_FLINT_LLL_CONFIGURATION,
        refresh=refresh,
    )
    if (
        runtime.availability is CapabilityProviderAvailability.AVAILABLE
        and runtime.version != PYTHON_FLINT_VERSION
    ):
        return _unavailable_runtime(
            provider="python-flint",
            install_tier=CapabilityInstallTier.T1,
            license_id="MIT AND LGPL-3.0-or-later",
            diagnostic=(
                "Python-FLINT is installed but does not match the pinned "
                f"{PYTHON_FLINT_VERSION} LLL profile."
            ),
        )
    if refresh and runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        try:
            flint = importlib.import_module("flint")
        except (ImportError, OSError):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic="The pinned Python-FLINT LLL runtime cannot be imported.",
            )
        if getattr(flint, "__FLINT_VERSION__", None) != (
            PYTHON_FLINT_HNF_FLINT_VERSION
        ):
            return _unavailable_runtime(
                provider="python-flint",
                install_tier=CapabilityInstallTier.T1,
                license_id="MIT AND LGPL-3.0-or-later",
                diagnostic=(
                    "Python-FLINT is installed but its linked FLINT library does "
                    "not match the pinned LLL profile."
                ),
            )
    return runtime


def lean_provider_runtime(
    *,
    profiles: Mapping[str, Mapping[str, Any]],
    checker_ids: tuple[str, ...],
) -> CapabilityProviderRuntime:
    """Inspect the separately managed pinned Lean/Mathlib runtime."""

    from jacobian_checkers import lean4

    require_mathlib = any(
        profile.get("mathlib_commit") is not None for profile in profiles.values()
    )
    try:
        executable, _ = lean4.inspect_runtime(require_mathlib=require_mathlib)
        digest = _sha256_file(executable)
    except (OSError, RuntimeError):
        return _unavailable_runtime(
            provider="jacobian.lean4",
            install_tier=CapabilityInstallTier.T3,
            license_id="Apache-2.0",
            diagnostic=(
                f"The pinned Lean {lean4.LEAN_VERSION} runtime is unavailable."
            ),
        )
    return CapabilityProviderRuntime(
        provider="jacobian.lean4",
        availability=CapabilityProviderAvailability.AVAILABLE,
        version=lean4.LEAN_VERSION,
        digest=digest,
        digest_kind=CapabilityProviderDigestKind.EXECUTABLE,
        platform=_platform_tag(),
        install_tier=CapabilityInstallTier.T3,
        license_id="Apache-2.0",
        features=tuple(sorted(profiles)),
        checker_ids=checker_ids,
        configuration={"profiles": dict(profiles)},
    )
