#!/usr/bin/env python3
"""Run non-executing static checks for repository-owned benchmark code.

The benchmark tree contains verifier and validation Python alongside ordinary
benchmark tooling.  Ruff scans the complete tree, including those boundary
directories.  Mypy checks the benchmark control scripts with the repository's
strict configuration while skipping imported implementation bodies; importing
or executing a task, verifier, Oracle, or model is not part of this gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    ToolCommandRequest,
    ToolCommandStatus,
    operator_environment,
    run_tool_command,
)

RUFF_TARGETS = ("benchmarks", "tools/check_benchmark_static.py")
MYPY_TARGETS = (
    "tools/check_benchmark_adapters.py",
    "tools/check_benchmark_contracts.py",
    "tools/check_benchmark_static.py",
)


def _commands() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (
            "Ruff lint",
            ("-m", "ruff", "check", *RUFF_TARGETS),
        ),
        (
            "Ruff format",
            ("-m", "ruff", "format", "--check", *RUFF_TARGETS),
        ),
        (
            "mypy",
            (
                "-m",
                "mypy",
                "--follow-imports=skip",
                *MYPY_TARGETS,
            ),
        ),
    )


def main() -> int:
    """Run every static check and stop at the first failed gate."""
    for label, arguments in _commands():
        result = run_tool_command(
            ToolCommandRequest(
                executable=sys.executable,
                arguments=arguments,
                environment=operator_environment(),
                cwd=str(ROOT),
                timeout_seconds=300.0,
                stdout_limit_bytes=4 * 1024 * 1024,
                stderr_limit_bytes=4 * 1024 * 1024,
            )
        )
        if result.status is not ToolCommandStatus.EXITED:
            detail = result.diagnostic or result.stderr.decode(errors="replace")[:1024]
            print(f"{label} could not start: {detail}", file=sys.stderr)
            return 1
        if result.exit_code:
            output = (result.stdout + result.stderr).decode(errors="replace").strip()
            if output:
                print(output, file=sys.stderr)
            print(f"{label} failed with exit code {result.exit_code}", file=sys.stderr)
            return int(result.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
