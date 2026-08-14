"""Run pytest with a unique worktree-local temporary tree and process-tree backstop."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.process_supervisor import run_process_tree  # noqa: E402

_SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class PytestResult:
    """Outcome of one supervised pytest process."""

    exit_code: int
    status: str
    actual_seconds: float
    basetemp: Path
    retained: bool


def _has_basetemp(arguments: Sequence[str]) -> bool:
    return any(
        argument == "--basetemp" or argument.startswith("--basetemp=")
        for argument in arguments
    )


def _unique_basetemp(root: Path, name: str) -> Path:
    label = _SAFE_LABEL.sub("-", name).strip("-") or "pytest"
    run_root = root / ".pytest_cache" / "basetemp" / f"{label}-{uuid.uuid4().hex}"
    return run_root / "pytest"


def run_pytest(
    arguments: Sequence[str],
    *,
    root: Path,
    name: str,
    environment: Mapping[str, str],
    timeout_seconds: float = 3600.0,
    retain_on_failure: bool = False,
) -> PytestResult:
    """Execute pytest and clean its unique temp tree unless retention is requested."""

    if _has_basetemp(arguments):
        raise ValueError("pytest basetemp is owned by tools.pytest_lifecycle")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    basetemp = _unique_basetemp(root.resolve(), name)
    basetemp.mkdir(parents=True)
    run_root = basetemp.parent
    result: PytestResult | None = None
    try:
        started = time.monotonic()
        tree = run_process_tree(
            (sys.executable, "-m", "pytest", *arguments, f"--basetemp={basetemp}"),
            timeout=timeout_seconds,
            cwd=root.resolve(),
            env=environment,
        )
        elapsed = time.monotonic() - started
        if tree.timed_out:
            print(
                f"[{name}] process tree timed out after {timeout_seconds}s",
                file=sys.stderr,
            )
        retained = bool(tree.exit_code and retain_on_failure)
        result = PytestResult(
            exit_code=tree.exit_code,
            status="TIMED_OUT" if tree.timed_out else "EXITED",
            actual_seconds=elapsed,
            basetemp=basetemp,
            retained=retained,
        )
    finally:
        if result is not None and result.retained:
            print(f"[{name}] retained failed pytest tree: {run_root}", file=sys.stderr)
        else:
            shutil.rmtree(run_root, ignore_errors=True)
    assert result is not None
    return result


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=os.environ.get("PYTEST_RUN_NAME", "pytest"))
    parser.add_argument(
        "--retain-on-failure",
        action="store_true",
        default=os.environ.get("PYTEST_RETAIN_ON_FAILURE") == "1",
    )
    parser.add_argument("--timeout-seconds", type=_positive_float, default=3600.0)
    parser.add_argument("pytest_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.pytest_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        parser.error("pytest arguments are required after --")
    try:
        result = run_pytest(
            arguments,
            root=ROOT,
            name=args.name,
            environment=dict(os.environ),
            timeout_seconds=args.timeout_seconds,
            retain_on_failure=args.retain_on_failure,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PytestResult", "main", "run_pytest"]
