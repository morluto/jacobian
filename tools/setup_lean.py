#!/usr/bin/env python3
"""Install the pinned Lean toolchain and build local Jacobian Lean targets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CommandRunner = Callable[[Sequence[str], Path], int]


def _run(arguments: Sequence[str], cwd: Path) -> int:
    completed = subprocess.run(list(arguments), cwd=cwd, check=False)
    return int(completed.returncode)


def _run_required(arguments: Sequence[str], *, cwd: Path, run: CommandRunner) -> None:
    status = run(arguments, cwd)
    if status:
        raise RuntimeError(f"command failed ({status}): {' '.join(arguments)}")


def setup_lean(repo: Path, *, run: CommandRunner = _run) -> None:
    """Install the repository-pinned Lean toolchain and build local targets."""

    repo = repo.resolve()
    if shutil.which("elan") is None:
        raise RuntimeError(
            "Lean setup requires elan; install it, then rerun `make setup-lean`"
        )
    toolchain = (repo / "lean" / "lean-toolchain").read_text(encoding="utf-8").strip()
    _run_required(("elan", "toolchain", "install", toolchain), cwd=repo, run=run)
    _run_required(("lake", "exe", "cache", "get"), cwd=repo / "lean", run=run)
    _run_required(
        ("lake", "build", "repl", "jacobian_lean_proof_state"),
        cwd=repo / "lean",
        run=run,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        setup_lean(args.repo)
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
