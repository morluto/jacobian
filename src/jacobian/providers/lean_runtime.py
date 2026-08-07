"""Provider-owned runtime declarations for Lean frontends and checkers."""

from collections.abc import Mapping
from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.provider_runtime import _platform_tag, _sha256_file, _unavailable_runtime


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
        lake_executable = (
            lean4.lake_launcher_path(executable) if require_mathlib else None
        )
        if require_mathlib and lake_executable is None:
            raise RuntimeError(
                "TOOLCHAIN_RESOLUTION: the pinned Lake launcher is unavailable"
            )
        lake_digest = (
            _sha256_file(lake_executable) if lake_executable is not None else None
        )
    except (OSError, RuntimeError):
        return _unavailable_runtime(
            provider="jacobian.lean4",
            install_tier=CapabilityInstallTier.T3,
            license_id="Apache-2.0",
            diagnostic=(
                f"The pinned Lean {lean4.LEAN_VERSION} runtime is unavailable."
            ),
        )
    configuration: dict[str, Any] = {
        "executable": str(executable),
        "profiles": dict(profiles),
    }
    if lake_executable is not None:
        configuration["lake_executable"] = str(lake_executable)
        configuration["lake_digest"] = lake_digest
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
        configuration=configuration,
    )


def lean_frontend_provider_runtime() -> CapabilityProviderRuntime:
    """Inspect the pinned Lean executable used by CORE elaboration capabilities."""

    from jacobian_checkers import lean4

    try:
        executable, _ = lean4.inspect_runtime(require_mathlib=False)
        digest = _sha256_file(executable)
    except (OSError, RuntimeError) as exc:
        return _unavailable_runtime(
            provider="jacobian.lean4",
            install_tier=CapabilityInstallTier.T3,
            license_id="Apache-2.0",
            diagnostic=str(exc)
            or f"The pinned Lean {lean4.LEAN_VERSION} executable is unavailable.",
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
        features=("CORE", "elaboration", "lean-statement"),
        configuration={
            "executable": str(executable),
            "profiles": {
                "CORE": {
                    "import_name": "Init.Prelude",
                    "lean_commit": lean4.LEAN_COMMIT,
                }
            },
        },
    )
