"""Provider-owned runtime declarations for Lean frontends and checkers."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.provider_runtime import _platform_tag, _sha256_file, _unavailable_runtime

_CORE_LIBRARY_MODULE = Path("lib/lean/Init.olean")
_MATHLIB_PROJECT_FILES = (
    Path("lake-manifest.json"),
    Path("lakefile.toml"),
    Path("lean-toolchain"),
)
_MATHLIB_SOURCE_MODULES = (
    Path("JacobianLeanRuntime.lean"),
    Path("JacobianLeanProofState.lean"),
)
_MATHLIB_LOADED_MODULES = (
    Path(".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean"),
    Path(".lake/build/lib/lean/JacobianLeanRuntime.olean"),
    Path(".lake/build/lib/lean/JacobianLeanProofState.olean"),
    Path(".lake/build/bin/jacobian_lean_proof_state"),
)


class LeanRuntimeIdentityError(RuntimeError):
    """The declared Lean semantic environment cannot be reproduced exactly."""


def lean_mathlib_git_config(project_root: Path) -> dict[str, str]:
    """Return process-local Git trust for exact manifest-owned checkouts.

    Lake consults Git even when it launches already-built binaries. Immutable
    deployments are deliberately owned by root, so an unprivileged service
    needs exact ``safe.directory`` entries without a persistent or wildcard
    Git configuration.
    """

    root = project_root.resolve(strict=True)
    manifest = json.loads((root / "lake-manifest.json").read_text(encoding="utf-8"))
    packages = manifest.get("packages")
    if manifest.get("packagesDir") != ".lake/packages" or not isinstance(
        packages, list
    ):
        raise LeanRuntimeIdentityError("the Lean package manifest is malformed")
    package_root = (root / ".lake" / "packages").resolve(strict=True)
    checkouts: list[str] = []
    for package in packages:
        if not isinstance(package, dict):
            raise LeanRuntimeIdentityError("the Lean package manifest is malformed")
        name = package.get("name")
        if (
            package.get("type") != "git"
            or not isinstance(name, str)
            or not name
            or Path(name).name != name
        ):
            raise LeanRuntimeIdentityError("the Lean package manifest is malformed")
        candidate = package_root / name
        if candidate.is_symlink():
            raise LeanRuntimeIdentityError("a Lean package checkout is not exact")
        checkout = candidate.resolve(strict=True)
        if checkout.parent != package_root or not checkout.is_dir():
            raise LeanRuntimeIdentityError("a Lean package checkout is not exact")
        checkouts.append(str(checkout))
    environment = {
        "GIT_CONFIG_COUNT": str(len(checkouts)),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for index, checkout_path in enumerate(checkouts):
        environment[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
        environment[f"GIT_CONFIG_VALUE_{index}"] = checkout_path
    return environment


def _identity_file(root: Path, relative_path: Path) -> dict[str, str]:
    candidate = root / relative_path
    try:
        resolved = candidate.resolve(strict=True)
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or not resolved.is_relative_to(root)
        ):
            raise OSError("not an exact regular file")
        digest = _sha256_file(resolved)
    except OSError as exc:
        raise LeanRuntimeIdentityError(
            f"the required Lean runtime component {relative_path.as_posix()} is unavailable"
        ) from exc
    return {"path": relative_path.as_posix(), "digest": digest}


def _identity_tree(root: Path, relative_path: Path) -> dict[str, object]:
    """Digest every imported Mathlib olean under one portable tree identity."""

    candidate = root / relative_path
    try:
        resolved = candidate.resolve(strict=True)
        if (
            not resolved.is_dir()
            or resolved.is_symlink()
            or not resolved.is_relative_to(root)
        ):
            raise OSError("not an exact directory")
        files = tuple(
            sorted(path for path in resolved.rglob("*.olean") if path.is_file())
        )
        if not files or any(path.is_symlink() for path in files):
            raise OSError("incomplete or symlinked module tree")
    except OSError as exc:
        raise LeanRuntimeIdentityError(
            f"the required Lean runtime module tree {relative_path.as_posix()} is unavailable"
        ) from exc
    entries = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "digest": _sha256_file(path),
        }
        for path in files
    ]
    return {
        "path": relative_path.as_posix(),
        "digest": "sha256:" + hashlib.sha256(canonicalize_json(entries)).hexdigest(),
        "file_count": len(entries),
    }


def lean_semantic_runtime_identity(
    *,
    executable: Path,
    mathlib_runtime: Path | None,
) -> dict[str, Any]:
    """Measure every file that fixes the Lean frontend's effective semantics.

    The Lean binary alone does not determine a Mathlib invocation: Lake project
    configuration selects dependency roots, and the loaded olean modules and
    proof-state helper are executable input. Paths are recorded relative to
    their resolved roots so the identity remains portable while every byte that
    contributes to the environment is bound.
    """

    try:
        resolved_executable = executable.resolve(strict=True)
        if not resolved_executable.is_file() or resolved_executable.is_symlink():
            raise OSError("not an exact regular file")
        executable_digest = _sha256_file(resolved_executable)
        toolchain_root = resolved_executable.parent.parent.resolve(strict=True)
    except OSError as exc:
        raise LeanRuntimeIdentityError(
            "the pinned Lean executable is unavailable for semantic identity"
        ) from exc

    identity: dict[str, Any] = {
        "contract": "jacobian.lean.semantic-runtime/v1",
        "executable": {
            "digest": executable_digest,
            "library_module": _identity_file(toolchain_root, _CORE_LIBRARY_MODULE),
        },
    }
    if mathlib_runtime is None:
        return identity

    try:
        project_root = mathlib_runtime.resolve(strict=True)
        if not project_root.is_dir() or project_root.is_symlink():
            raise OSError("not an exact project directory")
    except OSError as exc:
        raise LeanRuntimeIdentityError(
            "the pinned Lean Mathlib project is unavailable for semantic identity"
        ) from exc
    lake = resolved_executable.with_name(
        "lake.exe" if resolved_executable.suffix.lower() == ".exe" else "lake"
    )
    try:
        if not lake.is_file() or lake.is_symlink():
            raise OSError("not an exact Lake launcher")
        lake_digest = _sha256_file(lake)
    except OSError as exc:
        raise LeanRuntimeIdentityError(
            "the pinned Lake launcher is unavailable for semantic identity"
        ) from exc
    identity["mathlib_project"] = {
        "root": str(project_root),
        "lake_digest": lake_digest,
        "configuration": [
            _identity_file(project_root, path) for path in _MATHLIB_PROJECT_FILES
        ],
        "source_modules": [
            _identity_file(project_root, path) for path in _MATHLIB_SOURCE_MODULES
        ],
        "loaded_modules": [
            _identity_file(project_root, path) for path in _MATHLIB_LOADED_MODULES
        ],
        "mathlib_module_tree": _identity_tree(
            project_root,
            Path(".lake/packages/mathlib/.lake/build/lib/lean/Mathlib"),
        ),
    }
    return identity


def require_lean_semantic_runtime_identity(
    runtime: CapabilityProviderRuntime,
) -> None:
    """Fail closed unless the declared Lean semantic environment is unchanged."""

    configuration = runtime.configuration
    executable_value = configuration.get("executable")
    expected = configuration.get("semantic_runtime")
    if not isinstance(executable_value, str) or not isinstance(expected, dict):
        raise LeanRuntimeIdentityError(
            "the Lean semantic runtime identity is incomplete"
        )
    project = expected.get("mathlib_project")
    project_root = project.get("root") if isinstance(project, dict) else None
    if project_root is not None and not isinstance(project_root, str):
        raise LeanRuntimeIdentityError("the Lean project runtime identity is malformed")
    measured = lean_semantic_runtime_identity(
        executable=Path(executable_value),
        mathlib_runtime=Path(project_root) if project_root is not None else None,
    )
    if canonicalize_json(measured) != canonicalize_json(expected):
        raise LeanRuntimeIdentityError("the Lean semantic runtime identity changed")


def lean_semantic_runtime_digest(identity: Mapping[str, Any]) -> str:
    """Return the canonical digest of an already measured semantic environment."""

    return "sha256:" + hashlib.sha256(canonicalize_json(dict(identity))).hexdigest()


def lean_portable_semantic_runtime_digest(identity: Mapping[str, Any]) -> str:
    """Digest semantic runtime content without binding its deployment root."""

    portable = dict(identity)
    project = portable.get("mathlib_project")
    if isinstance(project, Mapping):
        portable["mathlib_project"] = {
            key: value for key, value in project.items() if key != "root"
        }
    return lean_semantic_runtime_digest(portable)


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
        executable, mathlib_runtime = lean4.inspect_runtime(
            require_mathlib=require_mathlib
        )
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
        semantic_runtime = lean_semantic_runtime_identity(
            executable=executable,
            mathlib_runtime=mathlib_runtime,
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
        "semantic_runtime": semantic_runtime,
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
        semantic_runtime = lean_semantic_runtime_identity(
            executable=executable,
            mathlib_runtime=None,
        )
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
            "semantic_runtime": semantic_runtime,
            "profiles": {
                "CORE": {
                    "import_name": "Init.Prelude",
                    "lean_version": lean4.LEAN_VERSION,
                    "lean_commit": lean4.LEAN_COMMIT,
                    "mathlib_commit": None,
                }
            },
        },
    )
