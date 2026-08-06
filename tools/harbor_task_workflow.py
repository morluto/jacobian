#!/usr/bin/env python3
"""Prepare and validate explicitly selected Harbor benchmark tasks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    ToolCommandRequest,
    ToolCommandStatus,
    operator_environment,
    run_tool_command,
)
from benchmarks.tooling.harbor_suite import (  # noqa: E402
    HarborSuiteError,
    get_suite,
    select_task_refs,
)

PLANNER_PATH = ROOT / ".github" / "scripts" / "plan-benchmarks"
PYTEST_ROOT = ROOT / ".pytest_cache" / "harbor-validation"
OUTPUT_LIMIT = 8 * 1024 * 1024
OPERATOR_ENVIRONMENT = (
    "PATH",
    "HOME",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_HOST",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
)


class TaskWorkflowError(RuntimeError):
    """A selected-task workflow contract was invalid or a stage failed."""


@dataclass(frozen=True, slots=True)
class HostValidation:
    """One planner-owned host validation selector."""

    name: str
    selector: str
    keyword: str


@dataclass(frozen=True, slots=True)
class TaskSelection:
    """Resolved dataset tasks and their planner-owned host validation."""

    dataset: str
    tasks: tuple[str, ...]
    task_paths: tuple[Path, ...]
    host_validations: tuple[HostValidation, ...]


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Elapsed time for one completed workflow stage."""

    label: str
    seconds: float


def _load_planner() -> ModuleType:
    spec = importlib.util.spec_from_loader(
        "harbor_task_workflow_planner",
        SourceFileLoader("harbor_task_workflow_planner", str(PLANNER_PATH)),
    )
    if spec is None or spec.loader is None:
        raise TaskWorkflowError(f"could not load benchmark planner: {PLANNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_host_validations(raw: str) -> tuple[HostValidation, ...]:
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaskWorkflowError(
            "benchmark planner returned invalid host matrix JSON"
        ) from exc
    if not isinstance(entries, list) or not entries:
        raise TaskWorkflowError("benchmark planner selected no host validation")

    result: list[HostValidation] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TaskWorkflowError("benchmark host matrix entries must be objects")
        name = entry.get("name")
        selector = entry.get("selector")
        keyword = entry.get("keyword", "")
        splits = entry.get("splits", 0)
        group = entry.get("group", 0)
        if (
            not isinstance(name, str)
            or not isinstance(selector, str)
            or not isinstance(keyword, str)
        ):
            raise TaskWorkflowError("benchmark host matrix strings are malformed")
        if splits != 0 or group != 0:
            raise TaskWorkflowError(
                "selected-task validation may not use sharded selectors"
            )
        selector_path = Path(selector)
        if selector_path.is_absolute() or ".." in selector_path.parts:
            raise TaskWorkflowError(f"unsafe benchmark host selector: {selector}")
        result.append(HostValidation(name=name, selector=selector, keyword=keyword))
    return tuple(result)


def resolve_selection(dataset: str, tasks: tuple[str, ...]) -> TaskSelection:
    """Resolve task membership and host selectors once through canonical owners."""
    if not dataset:
        raise TaskWorkflowError("DATASET is required")
    try:
        suite = get_suite(dataset)
        refs = select_task_refs(suite, tasks)
    except HarborSuiteError as exc:
        raise TaskWorkflowError(str(exc)) from exc

    changed_paths = []
    for ref in refs:
        verifier = ref.path / "tests" / "verifier.py"
        source = verifier if verifier.is_file() else ref.path / "task.toml"
        changed_paths.append(source.relative_to(ROOT).as_posix())

    planner = _load_planner()
    plan_function = getattr(planner, "plan", None)
    if not callable(plan_function):
        raise TaskWorkflowError("benchmark planner does not expose plan()")
    plan = plan_function(changed_paths, event="pull_request")
    if not isinstance(plan, dict):
        raise TaskWorkflowError("benchmark planner returned a malformed plan")
    raw_host_matrix = plan.get("benchmark-host-validation-matrix")
    if not isinstance(raw_host_matrix, str):
        raise TaskWorkflowError("benchmark planner omitted the host validation matrix")

    return TaskSelection(
        dataset=suite.id,
        tasks=tuple(ref.path.name for ref in refs),
        task_paths=tuple(ref.path for ref in refs),
        host_validations=_parse_host_validations(raw_host_matrix),
    )


def _run_checked(
    label: str,
    arguments: tuple[str, ...],
    *,
    timings: list[StageTiming],
    timeout_seconds: float,
    executable: str | None = None,
) -> None:
    print(f"[{label}] {' '.join(arguments)}", flush=True)
    started = time.monotonic()
    result = run_tool_command(
        ToolCommandRequest(
            # Keep an uvx-selected interpreter path intact. Resolving its symlink
            # escapes the ephemeral environment and loses Harbor dependencies.
            executable=executable or sys.executable,
            arguments=arguments,
            environment=operator_environment(include=OPERATOR_ENVIRONMENT),
            cwd=str(ROOT),
            timeout_seconds=timeout_seconds,
            stdout_limit_bytes=OUTPUT_LIMIT,
            stderr_limit_bytes=OUTPUT_LIMIT,
        )
    )
    elapsed = time.monotonic() - started
    output = (result.stdout + result.stderr).decode(errors="replace").strip()
    if output:
        print(output)
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        detail = result.diagnostic or f"status={result.status}, exit={result.exit_code}"
        raise TaskWorkflowError(f"{label} failed after {elapsed:.2f}s ({detail})")
    timings.append(StageTiming(label=label, seconds=elapsed))


def _uv_executable() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise TaskWorkflowError("uv is required to run repository validation")
    return uv


def _task_tree_files(selection: TaskSelection, *, root: Path) -> tuple[Path, ...]:
    files = {
        path
        for task_path in selection.task_paths
        for path in task_path.rglob("*")
        if path.is_file()
    }
    for task in selection.tasks:
        leaf = (
            root
            / "benchmarks"
            / "validation"
            / selection.dataset.replace("-", "_")
            / f"test_{task.replace('-', '_')}.py"
        )
        if leaf.is_file():
            files.add(leaf)
    return tuple(sorted(files))


def _snapshot(paths: tuple[Path, ...], *, root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        path
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    )


def _python_paths(selection: TaskSelection, *, root: Path) -> tuple[str, ...]:
    return tuple(
        path.relative_to(root).as_posix()
        for path in _task_tree_files(selection, root=root)
        if path.suffix == ".py"
    )


def prepare(selection: TaskSelection) -> tuple[str, ...]:
    """Format and synchronize only the explicitly selected task surface."""
    timings: list[StageTiming] = []
    before = _snapshot(_task_tree_files(selection, root=ROOT), root=ROOT)
    python_paths = _python_paths(selection, root=ROOT)
    if python_paths:
        _run_checked(
            "format",
            ("run", "--locked", "ruff", "format", *python_paths),
            timings=timings,
            timeout_seconds=120.0,
            executable=_uv_executable(),
        )
    after_format = _snapshot(_task_tree_files(selection, root=ROOT), root=ROOT)
    formatted = _changed_paths(before, after_format)

    _run_checked(
        "sync-public-contract",
        (
            "-m",
            "benchmarks.tooling.public_contract",
            "sync-dataset",
            "--dataset-root",
            f"benchmarks/datasets/{selection.dataset}",
            "--tasks",
            *selection.tasks,
        ),
        timings=timings,
        timeout_seconds=120.0,
    )
    _run_checked(
        "sync-verifier-checksum",
        (
            "tools/sync_harbor_verifier_support.py",
            "--dataset",
            selection.dataset,
            "--tasks",
            *selection.tasks,
        ),
        timings=timings,
        timeout_seconds=120.0,
    )
    after_sync = _snapshot(_task_tree_files(selection, root=ROOT), root=ROOT)
    generated = _changed_paths(after_format, after_sync)
    changed = _changed_paths(before, after_sync)

    print("\nPreparation changes:")
    print("  formatted: " + (", ".join(formatted) if formatted else "none"))
    print("  generated: " + (", ".join(generated) if generated else "none"))
    print("  all changed: " + (", ".join(changed) if changed else "none"))
    _print_timings(timings)
    return changed


def _fresh_oracle_evidence(
    selection: TaskSelection, task: str, *, started_ns: int
) -> tuple[str, str]:
    evidence_root = ROOT / "benchmarks" / "results" / f"{selection.dataset}-oracle"
    candidates = sorted(
        evidence_root.glob("*/oracle-evidence.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        if path.stat().st_mtime_ns < started_ns:
            continue
        try:
            payload: Any = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("dataset") != selection.dataset:
            continue
        task_entries = payload.get("tasks")
        if not isinstance(task_entries, list):
            continue
        for entry in task_entries:
            if isinstance(entry, dict) and entry.get("task") == task:
                digest = entry.get("digest")
                if isinstance(digest, str):
                    return digest, path.relative_to(ROOT).as_posix()
    raise TaskWorkflowError(
        f"Oracle produced no fresh evidence for {selection.dataset}/{task}"
    )


def _print_timings(timings: list[StageTiming]) -> None:
    print("\nStage timings:")
    for timing in timings:
        print(f"  {timing.label}: {timing.seconds:.2f}s")
    print(f"  total: {sum(timing.seconds for timing in timings):.2f}s")


def validate(selection: TaskSelection) -> tuple[tuple[str, str, str], ...]:
    """Run the complete selected-task gate without changing tracked sources."""
    timings: list[StageTiming] = []
    oracle_evidence: list[tuple[str, str, str]] = []
    PYTEST_ROOT.mkdir(parents=True, exist_ok=True)
    pytest_temp = PYTEST_ROOT / f"run-{time.time_ns()}"
    pytest_temp.mkdir()
    try:
        _run_checked(
            "static-quality",
            ("run", "--locked", "python", "tools/check_benchmark_static.py"),
            timings=timings,
            timeout_seconds=360.0,
            executable=_uv_executable(),
        )
        _run_checked(
            "contracts",
            (
                "tools/check_harbor_dataset.py",
                "--dataset",
                selection.dataset,
                "--tasks",
                *selection.tasks,
            ),
            timings=timings,
            timeout_seconds=300.0,
        )
        for host in selection.host_validations:
            arguments: tuple[str, ...] = (
                "run",
                "--locked",
                "pytest",
                "-n",
                "0",
                "--durations=10",
                f"--basetemp={pytest_temp / host.name}",
                host.selector,
            )
            if host.keyword:
                arguments += ("-k", host.keyword)
            _run_checked(
                f"host:{host.name}",
                arguments,
                timings=timings,
                timeout_seconds=600.0,
                executable=_uv_executable(),
            )
        for task in selection.tasks:
            started_ns = time.time_ns()
            make = shutil.which("make")
            if make is None:
                raise TaskWorkflowError("make is required to run the exact Oracle")
            _run_checked(
                f"oracle:{task}",
                (
                    "--no-print-directory",
                    "harbor-oracle-run",
                    f"DATASET={selection.dataset}",
                    f"TASKS={task}",
                ),
                timings=timings,
                timeout_seconds=1800.0,
                executable=make,
            )
            digest, evidence_path = _fresh_oracle_evidence(
                selection, task, started_ns=started_ns
            )
            oracle_evidence.append((task, digest, evidence_path))
    finally:
        shutil.rmtree(pytest_temp, ignore_errors=True)
        if PYTEST_ROOT.is_dir() and not any(PYTEST_ROOT.iterdir()):
            PYTEST_ROOT.rmdir()

    print("\nOracle evidence:")
    for task, digest, evidence_path in oracle_evidence:
        print(f"  {task}: digest={digest} evidence={evidence_path}")
    _print_timings(timings)
    return tuple(oracle_evidence)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--tasks", nargs="+", required=True)
    return parser


def main() -> int:
    """Run the selected preparation or validation workflow."""
    args = _parser().parse_args()
    try:
        selection = resolve_selection(args.dataset, tuple(args.tasks))
        if args.command == "prepare":
            prepare(selection)
        else:
            validate(selection)
    except TaskWorkflowError as exc:
        print(f"harbor task workflow failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
