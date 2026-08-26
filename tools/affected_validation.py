#!/usr/bin/env python3
"""Run the CI planner's affected local validation for the current branch."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ci_test_plan import TestPlan, build_plan  # noqa: E402
from tools.command_runner import (  # noqa: E402
    ToolCommandStatus,
    operator_environment,
    run_operator_command,
)

_STATIC_PREFIXES = ("src/", "tests/", "benchmarks/")
_VALIDATION_ENVIRONMENT = (
    "PATH",
    "AFFECTED_BASE",
    "PYTEST_ARGS",
    "PYTEST_DIAGNOSTIC_ARGS",
    "ALLOW_PARALLEL_VALIDATION",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "VIRTUAL_ENV",
)
_BOUNDARY_LANE_TIMEOUT_SECONDS = 4_800
_VALIDATION_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024


def _validation_environment() -> dict[str, str]:
    """Forward documented validation controls without ambient environment leakage."""

    return dict(operator_environment(include=_VALIDATION_ENVIRONMENT))


def _git(*arguments: str, repository: Path) -> str:
    """Return one bounded Git metadata query's decoded standard output."""

    result = run_operator_command(
        "git",
        arguments,
        cwd=repository,
        timeout_seconds=30.0,
        stdout_limit_bytes=4 * 1024 * 1024,
        stderr_limit_bytes=1024 * 1024,
        environment=_validation_environment(),
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        diagnostic = result.diagnostic or result.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"git {' '.join(arguments)} failed: {diagnostic.strip()}")
    return result.stdout.decode("utf-8", "strict").strip()


def changed_paths(*, base: str, repository: Path) -> tuple[str, str, tuple[str, ...]]:
    """Resolve the PR-like base/head pair plus staged, unstaged, and new paths."""

    base_revision = _git("rev-parse", "--verify", base, repository=repository)
    head_revision = _git("rev-parse", "--verify", "HEAD", repository=repository)
    committed_paths = _git(
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-only",
        f"{base_revision}...{head_revision}",
        repository=repository,
    )
    unstaged_paths = _git("diff", "--name-only", repository=repository)
    staged_paths = _git("diff", "--cached", "--name-only", repository=repository)
    untracked_paths = _git(
        "ls-files", "--others", "--exclude-standard", repository=repository
    )
    paths = tuple(
        sorted(
            {
                path
                for output in (
                    committed_paths,
                    unstaged_paths,
                    staged_paths,
                    untracked_paths,
                )
                for path in output.splitlines()
                if path
            }
        )
    )
    return base_revision, head_revision, paths


def _static_paths(paths: Iterable[str], *, repository: Path) -> tuple[str, ...]:
    """Keep local static evidence aligned with the repository-wide check roots."""

    return tuple(
        path
        for path in paths
        if path.endswith(".py")
        and path.startswith(_STATIC_PREFIXES)
        and (repository / path).is_file()
    )


def commands_for_plan(
    plan: TestPlan,
    *,
    paths: Iterable[str],
    repository: Path,
) -> tuple[tuple[str, ...], ...]:
    """Return the ordered local commands for one immutable planner result."""

    commands: list[tuple[str, ...]] = []
    static_paths = _static_paths(paths, repository=repository)
    if static_paths:
        rendered_paths = " ".join(static_paths)
        commands.extend(
            (
                ("make", "lint-scoped", f"PATHS={rendered_paths}"),
                ("make", "typecheck-scoped", f"PATHS={rendered_paths}"),
            )
        )
    if plan.run_math:
        command = ["make", "test-math"]
        # A deleted test is still a meaningful changed path, but cannot be a
        # pytest selector. Fall back to the complete math lane rather than
        # silently dropping the owner evidence that CI would require.
        selected_math_tests = tuple(
            path for path in plan.math_tests if (repository / path).exists()
        )
        if plan.math_tests and len(selected_math_tests) != len(plan.math_tests):
            selected_math_tests = ()
        if selected_math_tests:
            command.append(f"TESTS={' '.join(selected_math_tests)}")
        commands.append(tuple(command))
    if plan.run_catalog:
        commands.append(("make", "test-catalog"))
    if plan.run_catalog_examples:
        commands.append(
            ("make", "test-integration", "TESTS=tests/integration/catalog/")
        )
    commands.extend(("make", f"test-{lane}") for lane in plan.python_lanes)
    commands.extend(("make", f"test-{lane}") for lane in plan.boundary_lanes)
    if plan.run_singular:
        commands.append(("make", "test-singular"))
    return tuple(commands)


def _run(commands: Sequence[Sequence[str]], *, repository: Path) -> None:
    for command in commands:
        print("+", " ".join(command), flush=True)
        result = run_operator_command(
            command[0],
            command[1:],
            cwd=repository,
            timeout_seconds=(
                _BOUNDARY_LANE_TIMEOUT_SECONDS
                if len(command) > 1 and command[1] in {"test-process", "test-mcp"}
                else 30 * 60
            ),
            stdout_limit_bytes=_VALIDATION_OUTPUT_LIMIT_BYTES,
            stderr_limit_bytes=_VALIDATION_OUTPUT_LIMIT_BYTES,
            environment=_validation_environment(),
        )
        if result.stdout:
            sys.stdout.write(result.stdout.decode("utf-8", "replace"))
        if result.stderr:
            sys.stderr.write(result.stderr.decode("utf-8", "replace"))
        if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
            raise SystemExit(
                result.diagnostic or f"{' '.join(command)} failed with {result.status}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Git base revision to compare with HEAD and local changes (default: origin/main)",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=ROOT,
        help="repository root (default: this repository)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planner result and selected commands without running them",
    )
    arguments = parser.parse_args(argv)
    repository = arguments.repository.resolve()
    base, head, paths = changed_paths(base=arguments.base, repository=repository)
    plan = build_plan(
        event="pull_request",
        base_revision=base,
        head_revision=head,
        changed_paths=paths,
        repository=repository,
    )
    commands = commands_for_plan(plan, paths=paths, repository=repository)
    print(plan.as_json())
    for reason in plan.reasons:
        print(f"reason: {reason}")
    if plan.run_wheel:
        print("CI-only evidence: installed-wheel validation remains selected.")
    if arguments.dry_run:
        for command in commands:
            print("+", " ".join(command))
        return 0
    _run(commands, repository=repository)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
