"""Fail-closed runtime identity probes for capability providers."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import sysconfig
from collections.abc import Mapping
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.implementation import ImplementationError, package_source_digest

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


class ProviderRuntimeErrorCode(StrEnum):
    """Stable classifications for provider identity and readiness failures."""

    IDENTITY_INCOMPLETE = "IDENTITY_INCOMPLETE"
    MALFORMED_RUNTIME = "MALFORMED_RUNTIME"
    UNAVAILABLE = "UNAVAILABLE"
    IDENTITY_CHANGED = "IDENTITY_CHANGED"
    READINESS_FAILED = "READINESS_FAILED"


class ProviderRuntimeError(RuntimeError):
    """Raised when a required provider identity cannot be inspected safely."""

    def __init__(
        self,
        message: str,
        *,
        code: ProviderRuntimeErrorCode | None = None,
    ) -> None:
        self.code = code or _infer_provider_error_code(message)
        super().__init__(message[:512])


def _infer_provider_error_code(message: str) -> ProviderRuntimeErrorCode:
    lowered = message.casefold()
    if "incomplete" in lowered:
        return ProviderRuntimeErrorCode.IDENTITY_INCOMPLETE
    if "malformed" in lowered or "invalid" in lowered:
        return ProviderRuntimeErrorCode.MALFORMED_RUNTIME
    if "changed" in lowered:
        return ProviderRuntimeErrorCode.IDENTITY_CHANGED
    return ProviderRuntimeErrorCode.UNAVAILABLE


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
            "the installed Python distribution has no hashed RECORD manifest",
            code=ProviderRuntimeErrorCode.MALFORMED_RUNTIME,
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
        digest = package_source_digest("jacobian.capability_service:CapabilityService")
    except (importlib.metadata.PackageNotFoundError, ImplementationError) as exc:
        raise ProviderRuntimeError(
            "the Jacobian source runtime could not be identified",
            code=ProviderRuntimeErrorCode.UNAVAILABLE,
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
                f"the {distribution_name} import target is unavailable",
                code=ProviderRuntimeErrorCode.UNAVAILABLE,
            )
        distribution = importlib.metadata.distribution(distribution_name)
        digest = _distribution_record_digest(distribution)
    except ProviderRuntimeError as exc:
        raise ProviderRuntimeError(
            f"the {distribution_name} distribution identity is unavailable",
            code=exc.code,
        ) from exc
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ProviderRuntimeError(
            f"the {distribution_name} distribution identity is unavailable",
            code=ProviderRuntimeErrorCode.UNAVAILABLE,
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
        diagnostic=diagnostic[:512],
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


def _validated_provider_runtime(
    runtime: CapabilityProviderRuntime,
) -> CapabilityProviderRuntime:
    try:
        return CapabilityProviderRuntime.model_validate(runtime.model_dump(mode="json"))
    except (AttributeError, TypeError, ValidationError) as exc:
        raise ProviderRuntimeError(
            "provider runtime is malformed",
            code=ProviderRuntimeErrorCode.MALFORMED_RUNTIME,
        ) from exc


def _require_executable_identity(runtime: CapabilityProviderRuntime) -> None:
    executable = runtime.configuration.get("executable")
    if not isinstance(executable, str):
        raise ProviderRuntimeError(
            "provider executable identity is incomplete",
            code=ProviderRuntimeErrorCode.IDENTITY_INCOMPLETE,
        )
    try:
        measured = _sha256_file(Path(executable))
    except OSError as exc:
        raise ProviderRuntimeError(
            "provider executable is unavailable",
            code=ProviderRuntimeErrorCode.UNAVAILABLE,
        ) from exc
    if measured != runtime.digest:
        raise ProviderRuntimeError(
            "provider executable identity changed",
            code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
        )


def _require_source_identity(runtime: CapabilityProviderRuntime) -> None:
    entrypoint = runtime.configuration.get("entrypoint")
    if not isinstance(entrypoint, str):
        raise ProviderRuntimeError(
            "provider source identity is incomplete",
            code=ProviderRuntimeErrorCode.IDENTITY_INCOMPLETE,
        )
    try:
        measured_source = package_source_digest(entrypoint)
    except ImplementationError as exc:
        raise ProviderRuntimeError(
            "provider source is unavailable",
            code=ProviderRuntimeErrorCode.UNAVAILABLE,
        ) from exc
    if measured_source != runtime.digest:
        raise ProviderRuntimeError(
            "provider source identity changed",
            code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
        )


def _require_python_identity(runtime: CapabilityProviderRuntime) -> None:
    distribution = runtime.configuration.get("distribution")
    import_name = runtime.distribution_import_name
    if not isinstance(distribution, str) or import_name is None:
        raise ProviderRuntimeError(
            "Python distribution identity is incomplete",
            code=ProviderRuntimeErrorCode.IDENTITY_INCOMPLETE,
        )
    try:
        version, digest, _license_files = _inspect_python_distribution_identity(
            distribution,
            import_name,
            runtime.distribution_required_attributes,
        )
    except ProviderRuntimeError as exc:
        raise ProviderRuntimeError(
            "Python distribution identity is unavailable",
            code=exc.code,
        ) from exc
    if version != runtime.version or digest != runtime.digest:
        raise ProviderRuntimeError(
            "Python distribution identity changed",
            code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
        )


def _component_runtimes(
    runtime: CapabilityProviderRuntime,
) -> tuple[CapabilityProviderRuntime, ...]:
    components = runtime.configuration.get("components")
    if not isinstance(components, list) or not components:
        raise ProviderRuntimeError(
            "composite provider identity is incomplete",
            code=ProviderRuntimeErrorCode.IDENTITY_INCOMPLETE,
        )
    try:
        return tuple(
            CapabilityProviderRuntime.model_validate(component)
            for component in components
        )
    except ValidationError as exc:
        raise ProviderRuntimeError(
            "composite provider runtime is malformed",
            code=ProviderRuntimeErrorCode.MALFORMED_RUNTIME,
        ) from exc


def _require_composite_identity(runtime: CapabilityProviderRuntime) -> None:
    component_runtimes = _component_runtimes(runtime)
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
        raise ProviderRuntimeError(
            "composite provider identity changed",
            code=ProviderRuntimeErrorCode.IDENTITY_CHANGED,
        )


def require_provider_runtime_unchanged(
    runtime: CapabilityProviderRuntime,
) -> None:
    """Remeasure every immutable component of an authorized provider runtime."""

    runtime = _validated_provider_runtime(runtime)

    if (
        runtime.availability is not CapabilityProviderAvailability.AVAILABLE
        or runtime.digest is None
        or runtime.digest_kind is None
    ):
        raise ProviderRuntimeError(
            "provider runtime identity is incomplete",
            code=ProviderRuntimeErrorCode.IDENTITY_INCOMPLETE,
        )

    if runtime.digest_kind is CapabilityProviderDigestKind.EXECUTABLE:
        _require_executable_identity(runtime)
        return

    if runtime.digest_kind is CapabilityProviderDigestKind.SOURCE_TREE:
        _require_source_identity(runtime)
        return

    if runtime.digest_kind is CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD:
        _require_python_identity(runtime)
        return

    _require_composite_identity(runtime)


def require_provider_runtime_ready(runtime: CapabilityProviderRuntime) -> None:
    """Check first-use callable availability without changing identity policy."""

    runtime = _validated_provider_runtime(runtime)
    if runtime.availability is not CapabilityProviderAvailability.AVAILABLE:
        raise ProviderRuntimeError(
            "provider runtime is unavailable",
            code=ProviderRuntimeErrorCode.UNAVAILABLE,
        )
    if runtime.digest_kind is CapabilityProviderDigestKind.PYTHON_DISTRIBUTION_RECORD:
        _require_python_ready(runtime)
        return
    if runtime.digest_kind is CapabilityProviderDigestKind.EXECUTABLE:
        _require_executable_ready(runtime)
        return
    if runtime.digest_kind is CapabilityProviderDigestKind.COMPOSITE:
        for component in _component_runtimes(runtime):
            require_provider_runtime_ready(component)


def _require_python_ready(runtime: CapabilityProviderRuntime) -> None:
    import_name = runtime.distribution_import_name
    if import_name is None:
        raise ProviderRuntimeError(
            "Python provider readiness identity is incomplete",
            code=ProviderRuntimeErrorCode.IDENTITY_INCOMPLETE,
        )
    try:
        module = importlib.import_module(import_name)
        for attribute in runtime.distribution_required_attributes:
            getattr(module, attribute)
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        raise ProviderRuntimeError(
            "Python provider required attributes are not ready",
            code=ProviderRuntimeErrorCode.READINESS_FAILED,
        ) from exc


def _require_executable_ready(runtime: CapabilityProviderRuntime) -> None:
    executable = runtime.configuration.get("executable")
    if not isinstance(executable, str) or not Path(executable).is_file():
        raise ProviderRuntimeError(
            "provider executable is not ready",
            code=ProviderRuntimeErrorCode.READINESS_FAILED,
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
                    f"{Z3_SOLVER_VERSION} SMT solver profile."
                ),
            )
        return runtime
    return jacobian_provider_runtime(
        provider,
        features=features,
        checker_ids=checker_ids,
        configuration=configuration,
    )
