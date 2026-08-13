#!/usr/bin/env python3
"""Authoritative local development setup profiles and diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROFILE_DOCUMENTATION = "docs/how-to/setup-agent-from-source.md#profiles"
NATIVE_PROVIDER_DOCUMENTATION = "docs/how-to/install-native-and-formal-providers.md"


@dataclass(frozen=True, slots=True)
class DevelopmentProfile:
    """One supported, deterministic development environment."""

    name: str
    providers: tuple[str, ...]


PROFILES = {
    "core": DevelopmentProfile(
        "core",
        ("networkx", "sympy", "z3", "python-flint", "python-flint-hnf", "cvc5"),
    ),
    "lean": DevelopmentProfile(
        "lean",
        (
            "networkx",
            "sympy",
            "z3",
            "python-flint",
            "python-flint-hnf",
            "cvc5",
            "lean",
        ),
    ),
    "external-proof": DevelopmentProfile(
        "external-proof",
        (
            "networkx",
            "sympy",
            "z3",
            "python-flint",
            "python-flint-hnf",
            "cvc5",
            "cadical",
            "drat-trim",
            "carcara",
        ),
    ),
}
PROFILE_NAMES = tuple(PROFILES)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One read-only profile readiness observation."""

    requirement: str
    status: str
    expected: str
    found: str | None
    recovery: str
    documentation: str


CommandRunner = Callable[[Sequence[str], Path], int]


def profile(name: str) -> DevelopmentProfile:
    """Return a supported profile or fail with the complete choice list."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILE_NAMES)
        raise ValueError(f"unknown profile: {name}; choose one of: {choices}") from exc


def sync_arguments(name: str, *, development: bool = True) -> tuple[str, ...]:
    """Return locked uv sync arguments for a profile."""

    profile(name)
    return ("sync", "--locked", "--dev" if development else "--no-dev")


def _run(arguments: Sequence[str], cwd: Path) -> int:
    completed = subprocess.run(list(arguments), cwd=cwd, check=False)
    return int(completed.returncode)


def _tool_path(name: str) -> str | None:
    return shutil.which(name)


def _capture_stdout(
    arguments: Sequence[str], *, cwd: Path, timeout_seconds: float
) -> str | None:
    executable = arguments[0]
    resolved = (
        executable if Path(executable).is_absolute() else _tool_path(str(executable))
    )
    if resolved is None:
        return None
    try:
        completed = subprocess.run(
            [resolved, *arguments[1:]],
            cwd=cwd,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode(errors="replace")


def _run_required(arguments: Sequence[str], *, cwd: Path, run: CommandRunner) -> None:
    status = run(arguments, cwd)
    if status:
        raise RuntimeError(f"command failed ({status}): {' '.join(arguments)}")


def setup_profile(repo: Path, name: str, *, run: CommandRunner = _run) -> None:
    """Install one profile without updating dependency manifests."""

    selected = profile(name)
    repo = repo.resolve()
    uv_diagnostic = _uv_diagnostic(repo)
    if uv_diagnostic.status != "available":
        _print_diagnostics((uv_diagnostic,))
        raise RuntimeError(
            f"uv {uv_diagnostic.expected} is required before profile setup"
        )
    _run_required(("uv", *sync_arguments(name)), cwd=repo, run=run)
    prepare_profile(repo, name, run=run)
    # The doctor runs inside the newly synced environment. In particular, the
    # external-proof profile only probes operator-installed binaries; it never
    # downloads provenance-sensitive executables.
    _run_required(
        (
            "uv",
            "run",
            "--locked",
            "--no-sync",
            "python",
            str(repo / "tools" / "development_profiles.py"),
            "doctor",
            "--profile",
            selected.name,
            "--repo",
            str(repo),
        ),
        cwd=repo,
        run=run,
    )


def prepare_profile(repo: Path, name: str, *, run: CommandRunner = _run) -> None:
    """Perform profile-specific setup after a locked sync."""

    profile(name)
    repo = repo.resolve()
    if name == "lean":
        if _tool_path("elan") is None:
            raise RuntimeError(
                "the lean profile requires elan; install it, then rerun "
                "`make setup-lean`"
            )
        toolchain = (
            (repo / "lean" / "lean-toolchain").read_text(encoding="utf-8").strip()
        )
        _run_required(("elan", "toolchain", "install", toolchain), cwd=repo, run=run)
        _run_required(("lake", "exe", "cache", "get"), cwd=repo / "lean", run=run)
        _run_required(
            ("lake", "build", "repl", "jacobian_lean_proof_state"),
            cwd=repo / "lean",
            run=run,
        )


def _requirements(repo: Path) -> dict[str, str]:
    with (repo / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    values = list(pyproject["project"]["dependencies"])
    requirements: dict[str, str] = {}
    for value in values:
        match = re.match(r"([A-Za-z0-9_-]+)\s*(.*)", value)
        if match:
            requirements[match.group(1).lower()] = match.group(2)
    return requirements


def _version_parts(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _matches_spec(version: str, spec: str) -> bool:
    if not spec:
        return True
    actual = _version_parts(version)
    for clause in spec.split(","):
        clause = clause.strip()
        match = re.fullmatch(r"(==|>=|<=|>|<)(.+)", clause)
        if match is None:
            continue
        operator, expected_text = match.groups()
        expected = _version_parts(expected_text)
        comparisons = {
            "==": actual == expected,
            ">=": actual >= expected,
            "<=": actual <= expected,
            ">": actual > expected,
            "<": actual < expected,
        }
        if not comparisons[operator]:
            return False
    return True


def _distribution_diagnostic(
    *,
    requirement: str,
    distribution: str,
    expected: str,
    profile_name: str,
) -> Diagnostic:
    try:
        found = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        found = None
    status = (
        "unavailable"
        if found is None
        else "available"
        if _matches_spec(found, expected)
        else "incompatible"
    )
    return Diagnostic(
        requirement=requirement,
        status=status,
        expected=expected or "installed distribution",
        found=found,
        recovery="Run `make setup`",
        documentation=(
            NATIVE_PROVIDER_DOCUMENTATION
            if profile_name != "core"
            else PROFILE_DOCUMENTATION
        ),
    )


def _uv_diagnostic(repo: Path) -> Diagnostic:
    expected = (repo / ".uv-version").read_text(encoding="utf-8").strip()
    found: str | None = None
    output = _capture_stdout(("uv", "--version"), cwd=repo, timeout_seconds=30.0)
    if output is not None:
        parts = output.strip().split()
        found = parts[1] if len(parts) >= 2 else None
    status = (
        "available"
        if found == expected
        else "unavailable"
        if found is None
        else "incompatible"
    )
    return Diagnostic(
        requirement="uv",
        status=status,
        expected=expected,
        found=found,
        recovery=f"Install uv {expected}: https://docs.astral.sh/uv/",
        documentation=PROFILE_DOCUMENTATION,
    )


def _lean_diagnostics(repo: Path) -> list[Diagnostic]:
    expected = (repo / "lean" / "lean-toolchain").read_text(encoding="utf-8").strip()
    elan = _tool_path("elan")
    lake = _tool_path("lake")
    installed = False
    found: str | None = None
    output = _capture_stdout(
        ("elan", "toolchain", "list"), cwd=repo, timeout_seconds=30.0
    )
    if output is not None:
        toolchains = [
            line.split(maxsplit=1)[0] for line in output.splitlines() if line.strip()
        ]
        installed = expected in toolchains
        found = expected if installed else ", ".join(toolchains) or None
    return [
        Diagnostic(
            "elan",
            "available" if elan else "unavailable",
            "installed executable",
            elan,
            "Install elan, then run `make setup-lean`",
            PROFILE_DOCUMENTATION,
        ),
        Diagnostic(
            "Lean toolchain",
            "available"
            if installed and lake
            else "incompatible"
            if elan and lake
            else "unavailable",
            expected,
            found,
            "Run `make setup-lean`",
            PROFILE_DOCUMENTATION,
        ),
    ]


def _external_proof_diagnostics() -> list[Diagnostic]:
    # Imported after the locked environment exists; setup loads this module
    # before `uv sync` and must not import Jacobian at module import time.
    from jacobian.providers.external_solver_runtime import (
        CADICAL_VERSION,
        CARCARA_VERSION,
        DRAT_TRIM_RELEASE_TAG,
        cadical_provider_runtime,
        carcara_provider_runtime,
        drat_trim_provider_runtime,
    )

    diagnostics: list[Diagnostic] = []
    recovery = (
        "Install the pinned, operator-provenanced runtime; no download is automatic"
    )
    for executable, name, expected, resolve in (
        ("cadical", "CaDiCaL", CADICAL_VERSION, cadical_provider_runtime),
        ("drat-trim", "DRAT-trim", DRAT_TRIM_RELEASE_TAG, drat_trim_provider_runtime),
        ("carcara", "Carcara", CARCARA_VERSION, carcara_provider_runtime),
    ):
        runtime = resolve()
        available = runtime.availability.value == "AVAILABLE"
        diagnostics.append(
            Diagnostic(
                name,
                "available"
                if available
                else "incompatible"
                if _tool_path(executable)
                else "unavailable",
                expected,
                runtime.version,
                recovery,
                NATIVE_PROVIDER_DOCUMENTATION,
            )
        )
    return diagnostics


def inspect_profile(repo: Path, name: str) -> list[Diagnostic]:
    """Inspect one profile without writing the checkout or environment."""

    selected = profile(name)
    repo = repo.resolve()
    requirements = _requirements(repo)
    diagnostics = [_uv_diagnostic(repo)]
    for provider, distribution in (
        ("networkx", "networkx"),
        ("sympy", "sympy"),
        ("z3", "z3-solver"),
        ("python-flint", "python-flint"),
        ("python-flint-hnf", "python-flint"),
        ("cvc5", "cvc5"),
    ):
        if provider not in selected.providers:
            continue
        diagnostics.append(
            _distribution_diagnostic(
                requirement=provider,
                distribution=distribution,
                expected=requirements.get(distribution, ""),
                profile_name=name,
            )
        )
    if name == "lean":
        diagnostics.extend(_lean_diagnostics(repo))
    if name == "external-proof":
        diagnostics.extend(_external_proof_diagnostics())
    return diagnostics


def _print_diagnostics(
    diagnostics: Sequence[Diagnostic], *, json_output: bool = False
) -> None:
    if json_output:
        print(json.dumps([asdict(item) for item in diagnostics], indent=2))
        return
    for item in diagnostics:
        print(
            f"{item.status:12} {item.requirement}: expected {item.expected}; "
            f"found {item.found or 'not found'}"
        )
        if item.status != "available":
            print(f"  recovery: {item.recovery}")
            print(f"  docs: {item.documentation}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("validate", "sync-flags", "prepare", "setup", "doctor")
    )
    parser.add_argument("--profile", default="core")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--no-dev", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        profile(args.profile)
        if args.action == "validate":
            return 0
        if args.action == "sync-flags":
            print("\n".join(sync_arguments(args.profile, development=not args.no_dev)))
            return 0
        if args.action == "setup":
            setup_profile(args.repo, args.profile)
            return 0
        if args.action == "prepare":
            prepare_profile(args.repo, args.profile)
            return 0
        diagnostics = inspect_profile(args.repo, args.profile)
        _print_diagnostics(diagnostics, json_output=args.json)
        return 0 if all(item.status == "available" for item in diagnostics) else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
