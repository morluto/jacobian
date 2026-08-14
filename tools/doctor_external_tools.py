#!/usr/bin/env python3
"""Diagnose operator-installed tools that uv does not own."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jacobian.providers.external_solver_runtime import (  # noqa: E402
    CADICAL_VERSION,
    CARCARA_VERSION,
    DRAT_TRIM_RELEASE_TAG,
    cadical_provider_runtime,
    carcara_provider_runtime,
    drat_trim_provider_runtime,
)

LEAN_DOCUMENTATION = "docs/how-to/install-native-and-formal-providers.md"
NATIVE_PROVIDER_DOCUMENTATION = "docs/how-to/install-native-and-formal-providers.md"
REQUIREMENTS = ("lean", "external-proof")


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One read-only observation of an operator-installed executable."""

    requirement: str
    status: str
    expected: str
    found: str | None
    recovery: str
    documentation: str


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


def lean_diagnostics(repo: Path) -> list[Diagnostic]:
    """Inspect elan and the repository-pinned Lean toolchain."""

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
            LEAN_DOCUMENTATION,
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
            LEAN_DOCUMENTATION,
        ),
    ]


def external_proof_diagnostics() -> list[Diagnostic]:
    """Inspect pinned CaDiCaL, DRAT-trim, and Carcara executables."""

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


def inspect_external_tools(
    repo: Path, *, require: str | None = None
) -> list[Diagnostic]:
    """Inspect Lean and SAT proof tools; ``require`` selects a failing subset."""

    repo = repo.resolve()
    if require == "lean":
        return lean_diagnostics(repo)
    if require == "external-proof":
        return external_proof_diagnostics()
    return [*lean_diagnostics(repo), *external_proof_diagnostics()]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--require", choices=REQUIREMENTS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        diagnostics = inspect_external_tools(args.repo, require=args.require)
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 2
    _print_diagnostics(diagnostics, json_output=args.json)
    if args.require is None:
        return 0
    return 0 if all(item.status == "available" for item in diagnostics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
