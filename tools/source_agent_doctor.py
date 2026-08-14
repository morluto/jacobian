#!/usr/bin/env python3
"""Audit a source-bound Jacobian agent installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    ToolCommandStatus,
    run_operator_command,
)

import jacobian  # noqa: E402
from jacobian.canonical import canonicalize_json  # noqa: E402
from jacobian.contracts.operations import (  # noqa: E402
    OperationCatalogSnapshot,
    ProviderAvailability,
)
from jacobian.operation_catalog import (  # noqa: E402
    OperationCatalog as CompiledOperationCatalog,
)
from jacobian.operation_visibility import OperationVisibilityPolicy  # noqa: E402
from jacobian.persistence.migrations import (  # noqa: E402
    CURRENT_STATE_FORMAT_REVISION,
    STATE_MIGRATIONS,
    SUPPORTED_STATE_FLOOR,
)
from jacobian.persistence.state_health import (  # noqa: E402
    StateHealth,
    inspect_state_health,
)
from jacobian.provider_runtime import known_provider_runtime  # noqa: E402
from jacobian.providers.external_solver_runtime import (  # noqa: E402
    cadical_provider_runtime,
    carcara_provider_runtime,
    cvc5_provider_runtime,
    drat_trim_provider_runtime,
)
from jacobian.providers.flint_runtime import (  # noqa: E402
    python_flint_hnf_provider_runtime,
    python_flint_provider_runtime,
)
from jacobian.providers.lean_runtime import lean_frontend_provider_runtime  # noqa: E402
from jacobian.providers.sympy_runtime import (  # noqa: E402
    sympy_polynomial_normalization_provider_runtime,
)

PROFILE_NAMES = ("core", "lean", "external-proof")
_CORE_PROVIDERS = (
    "networkx",
    "sympy",
    "z3",
    "python-flint",
    "python-flint-hnf",
    "cvc5",
)
_PROFILE_PROVIDERS = {
    "core": _CORE_PROVIDERS,
    "lean": (*_CORE_PROVIDERS, "lean"),
    "external-proof": (*_CORE_PROVIDERS, "cadical", "drat-trim", "carcara"),
}


def _catalog_digest(catalog: OperationCatalogSnapshot) -> str:
    payload = {
        "catalog_version": catalog.catalog_version,
        "operations": [
            descriptor.model_dump(mode="json") for descriptor in catalog.operations
        ],
    }
    return f"sha256:{hashlib.sha256(canonicalize_json(payload)).hexdigest()}"


def _git(repo: Path, *args: str) -> str:
    result = run_operator_command(
        "git",
        ("-C", str(repo), *args),
        cwd=repo,
        timeout_seconds=30.0,
        stdout_limit_bytes=4 * 1024 * 1024,
        stderr_limit_bytes=4096,
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        diagnostic = result.diagnostic or result.stderr.decode(errors="replace")[:1024]
        raise RuntimeError(f"git {' '.join(args)} failed: {diagnostic}")
    return result.stdout.decode("utf-8", errors="strict").strip()


def _repository_version(repo: Path) -> str:
    """Return uv's normalized version for the selected project."""

    result = run_operator_command(
        "uv",
        ("version", "--project", str(repo), "--short"),
        cwd=repo,
        timeout_seconds=30.0,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        diagnostic = result.diagnostic or result.stderr.decode(errors="replace")[:1024]
        raise RuntimeError(f"uv version failed: {diagnostic}")
    return result.stdout.decode("utf-8", errors="strict").strip()


def _provider_report() -> dict[str, dict[str, Any]]:
    resolvers: dict[str, Callable[[], Any]] = {
        "cadical": cadical_provider_runtime,
        "carcara": carcara_provider_runtime,
        "cvc5": cvc5_provider_runtime,
        "drat-trim": drat_trim_provider_runtime,
        "sympy": sympy_polynomial_normalization_provider_runtime,
        "lean": lean_frontend_provider_runtime,
        "python-flint": python_flint_provider_runtime,
        "python-flint-hnf": python_flint_hnf_provider_runtime,
    }
    report: dict[str, dict[str, Any]] = {}
    for name, resolve in resolvers.items():
        provider_runtime = resolve()
        report[name] = {
            "availability": provider_runtime.availability.value,
            "provider": provider_runtime.provider,
            "version": provider_runtime.version,
            "digest": provider_runtime.digest,
            "digest_kind": (
                provider_runtime.digest_kind.value
                if provider_runtime.digest_kind is not None
                else None
            ),
            "diagnostic": provider_runtime.diagnostic,
        }
    for name, provider in (("networkx", "jacobian.networkx"), ("z3", "jacobian.z3")):
        provider_runtime = known_provider_runtime(provider)
        report[name] = {
            "availability": provider_runtime.availability.value,
            "provider": provider_runtime.provider,
            "version": provider_runtime.version,
            "digest": provider_runtime.digest,
            "digest_kind": (
                provider_runtime.digest_kind.value
                if provider_runtime.digest_kind is not None
                else None
            ),
            "diagnostic": provider_runtime.diagnostic,
        }
    return dict(sorted(report.items()))


def inspect_installation(
    *,
    repo: Path,
    state_dir: Path,
    profile: str,
    expected_revision: str,
    launcher_provider_path: str,
    launcher_project_environment: str,
    launcher_elan_home: str,
    launcher_lean_runtime: str,
) -> dict[str, Any]:
    """Return a source, catalog, and provider identity report."""

    repo = repo.resolve(strict=True)
    state_dir = state_dir.resolve()
    revision = _git(repo, "rev-parse", "HEAD")
    dirty = bool(_git(repo, "status", "--porcelain"))
    package_source = Path(jacobian.__file__).resolve()
    expected_source = (repo / "src" / "jacobian").resolve()
    source_matches = package_source.is_relative_to(expected_source)
    with (repo / "pyproject.toml").open("rb") as stream:
        declared_version = tomllib.load(stream)["project"]["version"]
    expected_version = _repository_version(repo)

    state_health = inspect_state_health(
        state_dir,
        STATE_MIGRATIONS,
        supported_floor=SUPPORTED_STATE_FLOOR,
        current_revision=CURRENT_STATE_FORMAT_REVISION,
    )
    if state_health.blocking:
        return _state_incompatible_report(
            repo=repo,
            state_dir=state_dir,
            profile=profile,
            expected_revision=expected_revision,
            revision=revision,
            expected_version=expected_version,
            declared_version=declared_version,
            package_source=package_source,
            dirty=dirty,
            source_matches=source_matches,
            state_health=state_health,
        )

    compiled_catalog = CompiledOperationCatalog(
        state_dir / "metadata.sqlite3",
        OperationVisibilityPolicy(),
        expected_package_version=jacobian.__version__,
    )
    catalog = compiled_catalog.snapshot()
    providers = _provider_report()
    digest = _catalog_digest(catalog)
    diagnostics = list(compiled_catalog.header.diagnostics)

    missing = [
        provider
        for provider in _PROFILE_PROVIDERS[profile]
        if providers.get(provider, {}).get("availability")
        != ProviderAvailability.AVAILABLE.value
    ]
    effective_provider_path = os.environ.get("PATH", "")
    effective_project_environment = os.environ.get("UV_PROJECT_ENVIRONMENT", "")
    effective_elan_home = os.environ.get("ELAN_HOME", "")
    effective_lean_runtime = os.environ.get("JACOBIAN_LEAN_RUNTIME", "")
    checks = {
        "git_clean": not dirty,
        "revision_matches": revision == expected_revision,
        "package_version_matches": jacobian.__version__ == expected_version,
        "source_checkout_matches": source_matches,
        "profile_providers_available": not missing,
        "state_compatible": not state_health.blocking,
        "provider_path_preserved": effective_provider_path == launcher_provider_path
        or effective_provider_path.endswith(os.pathsep + launcher_provider_path),
        "project_environment_preserved": (
            effective_project_environment == launcher_project_environment
        ),
        "elan_home_preserved": effective_elan_home == launcher_elan_home,
        "lean_runtime_preserved": (effective_lean_runtime == launcher_lean_runtime),
    }
    return {
        "status": "ok" if all(checks.values()) else "error",
        "profile": profile,
        "repo": str(repo),
        "state_dir": str(state_dir),
        "git_revision": revision,
        "expected_git_revision": expected_revision,
        "git_dirty": dirty,
        "package_version": jacobian.__version__,
        "expected_package_version": expected_version,
        "declared_package_version": declared_version,
        "package_source": str(package_source),
        "provider_path": effective_provider_path,
        "launcher_provider_path": launcher_provider_path,
        "project_environment": effective_project_environment,
        "launcher_project_environment": launcher_project_environment,
        "elan_home": effective_elan_home,
        "launcher_elan_home": launcher_elan_home,
        "lean_runtime": effective_lean_runtime,
        "launcher_lean_runtime": launcher_lean_runtime,
        "catalog_digest": digest,
        "catalog_size": len(catalog.operations),
        "policy_profile": catalog.policy_profile,
        "policy_digest": catalog.policy_digest,
        "providers": providers,
        "missing_profile_providers": missing,
        "catalog_diagnostics": diagnostics,
        "state_health": state_health.as_dict(),
        "checks": checks,
    }


def _state_incompatible_report(
    *,
    repo: Path,
    state_dir: Path,
    profile: str,
    expected_revision: str,
    revision: str,
    expected_version: str,
    declared_version: str,
    package_source: Path,
    dirty: bool,
    source_matches: bool,
    state_health: StateHealth,
) -> dict[str, Any]:
    """Return a useful report without attempting to mutate incompatible state."""

    return {
        "status": "error",
        "profile": profile,
        "repo": str(repo),
        "state_dir": str(state_dir),
        "git_revision": revision,
        "expected_git_revision": expected_revision,
        "git_dirty": dirty,
        "package_version": jacobian.__version__,
        "expected_package_version": expected_version,
        "declared_package_version": declared_version,
        "package_source": str(package_source),
        "provider_path": os.environ.get("PATH", ""),
        "launcher_provider_path": "",
        "project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT", ""),
        "launcher_project_environment": "",
        "elan_home": os.environ.get("ELAN_HOME", ""),
        "launcher_elan_home": "",
        "lean_runtime": os.environ.get("JACOBIAN_LEAN_RUNTIME", ""),
        "launcher_lean_runtime": "",
        "catalog_digest": None,
        "catalog_size": 0,
        "policy_profile": None,
        "policy_digest": None,
        "providers": {},
        "missing_profile_providers": [],
        "catalog_diagnostics": [],
        "state_health": state_health.as_dict(),
        "checks": {
            "git_clean": not dirty,
            "revision_matches": revision == expected_revision,
            "package_version_matches": jacobian.__version__ == expected_version,
            "source_checkout_matches": source_matches,
            "state_compatible": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--provider-path", required=True)
    parser.add_argument("--project-environment", default="")
    parser.add_argument("--elan-home", default="")
    parser.add_argument("--lean-runtime", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = inspect_installation(
            repo=args.repo,
            state_dir=args.state_dir,
            profile=args.profile,
            expected_revision=args.expected_revision,
            launcher_provider_path=args.provider_path,
            launcher_project_environment=args.project_environment,
            launcher_elan_home=args.elan_home,
            launcher_lean_runtime=args.lean_runtime,
        )
    except (
        OSError,
        RuntimeError,
        KeyError,
        ValueError,
    ) as error:
        print(f"source-agent doctor failed: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.json:
        print(rendered, end="")
    else:
        marker = "✓" if report["status"] == "ok" else "✗"
        print(f"{marker} source checkout: {report['git_revision']}")
        print(f"{marker} package: {report['package_version']}")
        state_health = report.get("state_health", {})
        state_status = state_health.get("status", "UNKNOWN")
        state_marker = "✗" if state_health.get("blocking", False) else "✓"
        print(f"{state_marker} state: {state_status}")
        if state_health.get("blocking", False):
            diagnostic = state_health.get("diagnostic")
            if diagnostic:
                print(f"  {diagnostic}", file=sys.stderr)
            for mismatch in state_health.get("mismatches", ()):
                print(
                    "  migration {revision} ({name}) checksum differs".format(
                        **mismatch
                    ),
                    file=sys.stderr,
                )
            print(
                "  Preserve this state directory and use a compatible checkout "
                "to export it, or choose a fresh state directory; do not edit "
                "metadata.sqlite3.",
                file=sys.stderr,
            )
        print(
            f"{marker} catalog: {report['catalog_digest']} "
            f"({report['catalog_size']} operations)"
        )
        for provider, status in report["providers"].items():
            available = status["availability"] == "AVAILABLE"
            print(
                f"{'✓' if available else '-'} provider {provider}: "
                f"{status['availability']} ({status['version'] or 'not installed'})"
            )
        if report["missing_profile_providers"]:
            print(
                "missing providers for profile: "
                + ", ".join(report["missing_profile_providers"]),
                file=sys.stderr,
            )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
