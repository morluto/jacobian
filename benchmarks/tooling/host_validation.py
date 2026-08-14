"""Timing-aware orchestration and receipts for benchmark host validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from tools.benchmark_plan.model import plan_from_mapping
from tools.benchmark_plan.validation import validate_plan

from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.receipts import canonical_json, digest_bytes, receipt_digest
from benchmarks.tooling.validation_plan import (
    HostValidation,
    full_host_validation,
)

ROOT = Path(__file__).resolve().parents[2]
TIMING_PATH = ROOT / ".ci" / "benchmark-test-durations.json"
_PYTEST_RANDOMLY_SHARD_SEED = 0
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA = re.compile(r"[0-9a-f]{40,64}\Z")
_HOST_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
_HOST_KEYWORD = re.compile(r"[A-Za-z0-9_.-]*\Z")
_MAX_TIMING_BYTES = 5 * 1024 * 1024
_MAX_TIMING_ENTRIES = 10_000


def _shard_seed() -> int:
    """Return the pinned pytest-randomly seed shared by every host shard."""

    return _PYTEST_RANDOMLY_SHARD_SEED


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
    """Immutable identities shared by every shard in one host-validation run."""

    plan_head_sha: str
    execution_sha: str
    planner_digest: str
    topology_digest: str
    plan_digest: str


@dataclass(frozen=True, slots=True)
class ShardResult:
    """Normalized result returned by the pytest lifecycle boundary."""

    status: str
    exit_code: int
    actual_seconds: float


ShardRunner = Callable[[HostValidation, tuple[str, ...], int], ShardResult]


def _canonical(value: object) -> bytes:
    return cast(bytes, canonical_json(value))


def _digest_bytes(value: bytes) -> str:
    return cast(str, digest_bytes(value))


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise HarborSuiteError(f"{label} must be a sha256 digest")
    return value


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise HarborSuiteError(f"{label} must be a Git revision")
    return value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborSuiteError(f"{label} is unavailable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise HarborSuiteError(f"{label} must contain a JSON object")
    return value


def _entry(value: object) -> HostValidation:
    if not isinstance(value, Mapping) or set(value) != {
        "name",
        "selector",
        "keyword",
        "splits",
        "group",
        "predicted_seconds",
    }:
        raise HarborSuiteError("host-validation entry has an invalid shape")
    name = value["name"]
    selector = value["selector"]
    keyword = value["keyword"]
    splits = value["splits"]
    group = value["group"]
    predicted = value["predicted_seconds"]
    if (
        not isinstance(name, str)
        or not isinstance(selector, str)
        or not isinstance(keyword, str)
        or not isinstance(splits, int)
        or isinstance(splits, bool)
        or not isinstance(group, int)
        or isinstance(group, bool)
        or not isinstance(predicted, (int, float))
        or isinstance(predicted, bool)
        or not math.isfinite(float(predicted))
    ):
        raise HarborSuiteError("host-validation entry has invalid values")
    entry = HostValidation(
        name=name,
        selector=selector,
        keyword=keyword,
        splits=splits,
        group=group,
        predicted_seconds=float(predicted),
    )
    if (
        not entry.name
        or entry.name in {".", ".."}
        or Path(entry.name).name != entry.name
        or _HOST_NAME.fullmatch(entry.name) is None
        or (
            entry.selector != "benchmarks/validation"
            and not entry.selector.startswith("benchmarks/validation/")
        )
        or ".." in Path(entry.selector).parts
        or _HOST_KEYWORD.fullmatch(entry.keyword) is None
        or entry.predicted_seconds <= 0
        or (entry.splits == 0 and entry.group != 0)
        or (entry.splits > 0 and not 1 <= entry.group <= entry.splits)
        or entry.splits < 0
    ):
        raise HarborSuiteError("host-validation entry is incoherent")
    return entry


def load_plan(
    path: Path, *, execution_sha: str
) -> tuple[ExecutionProvenance, tuple[HostValidation, ...]]:
    """Validate a canonical plan.json and return its host matrix and provenance."""

    payload = _read_json_object(path, "benchmark plan")
    try:
        validate_plan(payload)
    except ValueError as exc:
        raise HarborSuiteError(f"benchmark plan is invalid: {exc}") from exc
    plan = plan_from_mapping(payload)
    entries = tuple(_entry(value) for value in plan.host_matrix)
    plan_digest = _digest_bytes(_canonical(plan.to_json_dict()))
    topology = plan.topology_digest
    if topology == "" and not entries and not plan.run_check:
        topology_digest = ""
    else:
        topology_digest = _require_digest(topology, "topology digest")
    head_sha = plan.head_sha or execution_sha
    return (
        ExecutionProvenance(
            plan_head_sha=_require_sha(head_sha, "plan head SHA"),
            execution_sha=_require_sha(execution_sha, "execution SHA"),
            planner_digest=_require_digest(plan.planner_digest, "planner digest"),
            topology_digest=topology_digest,
            plan_digest=plan_digest,
        ),
        entries,
    )


def local_provenance(
    root: Path, entries: Sequence[HostValidation]
) -> ExecutionProvenance:
    """Bind a local run to HEAD, planner sources, and the selected matrix."""

    from benchmarks.tooling.command_runner import git_head_sha

    execution_sha = git_head_sha(root)
    if execution_sha is None:
        raise HarborSuiteError("cannot resolve the local source revision")
    planner_sources = (
        root / ".github/scripts/plan-benchmarks",
        root / "benchmarks/tooling/validation_plan.py",
    )
    planner_digest = _digest_bytes(
        b"\n".join(path.read_bytes() for path in planner_sources)
    )
    matrix = [entry.as_matrix_entry() for entry in entries]
    plan_digest = _digest_bytes(_canonical(matrix))
    return ExecutionProvenance(
        plan_head_sha=execution_sha,
        execution_sha=execution_sha,
        planner_digest=planner_digest,
        topology_digest=plan_digest,
        plan_digest=plan_digest,
    )


def verify_execution_sha(root: Path, expected: str) -> str:
    """Require the checked-out tree to match the workflow execution identity."""

    expected = _require_sha(expected, "execution SHA")
    from benchmarks.tooling.command_runner import git_head_sha

    if git_head_sha(root) != expected:
        raise HarborSuiteError(
            "checked-out Git revision does not match --execution-sha"
        )
    return expected


def worker_allocation(
    *, entry_count: int, total_worker_budget: int, max_parallel: int
) -> tuple[int, int]:
    """Return concurrent shard processes and xdist workers per process.

    ``max_parallel`` reserves the process slots that may run concurrently,
    including slots executed by separate CI matrix jobs. Dividing the total
    budget by those slots prevents each under-filled invocation from consuming
    the complete budget as nested xdist workers.
    """

    if entry_count < 1:
        raise ValueError("entry_count must be positive")
    if total_worker_budget < 1:
        raise ValueError("total_worker_budget must be positive")
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    budgeted_slots = min(total_worker_budget, max_parallel)
    parallel = min(entry_count, budgeted_slots)
    workers = max(1, total_worker_budget // budgeted_slots)
    if budgeted_slots * workers > total_worker_budget:
        raise AssertionError("worker allocation exceeded its declared budget")
    return parallel, workers


def pytest_arguments(
    entry: HostValidation,
    *,
    timing_path: Path,
    workers: int,
    store_durations: bool,
) -> tuple[str, ...]:
    """Build one structured pytest argument vector from a validated entry."""

    if workers < 1:
        raise ValueError("workers must be positive")
    arguments = ["-n", str(workers), "--durations=10", entry.selector]
    if entry.keyword:
        arguments.extend(("-k", entry.keyword))
    arguments.extend(("--randomly-seed", str(_shard_seed())))
    if entry.splits:
        arguments.extend(
            (
                "--splits",
                str(entry.splits),
                "--group",
                str(entry.group),
                "--splitting-algorithm",
                "least_duration",
                "--durations-path",
                str(timing_path),
            )
        )
        if store_durations:
            arguments.extend(("--store-durations", "--clean-durations"))
    return tuple(arguments)


def _command_digest(
    entry: HostValidation, *, workers: int, store_durations: bool
) -> str:
    arguments = pytest_arguments(
        entry,
        timing_path=Path("<timing-input>"),
        workers=workers,
        store_durations=store_durations,
    )
    return cast(str, receipt_digest({"arguments": arguments}))


def timing_digest(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise HarborSuiteError(f"timing input is unavailable: {path}") from exc
    if len(payload) > _MAX_TIMING_BYTES:
        raise HarborSuiteError("timing input exceeds its byte limit")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HarborSuiteError("timing input is not valid JSON") from exc
    if not isinstance(value, dict) or len(value) > _MAX_TIMING_ENTRIES:
        raise HarborSuiteError("timing input has an invalid shape")
    for nodeid, duration in value.items():
        if (
            not isinstance(nodeid, str)
            or not nodeid.startswith("benchmarks/validation/")
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
        ):
            raise HarborSuiteError("timing input contains an invalid entry")
    return _digest_bytes(payload)


@contextmanager
def timing_input(root: Path, requested: Path) -> Iterator[tuple[Path, str]]:
    """Yield a validated timing file, using a worktree-local empty fallback."""

    run_root: Path | None = None
    path = requested
    try:
        requested_digest = timing_digest(requested)
    except HarborSuiteError as exc:
        run_root = root / ".pytest_cache" / "benchmark-host" / uuid.uuid4().hex
        run_root.mkdir(parents=True)
        path = run_root / "benchmark-test-durations.json"
        path.write_text("{}\n", encoding="utf-8")
        requested_digest = timing_digest(path)
        print(
            json.dumps(
                {
                    "event": "benchmark-host-timing-fallback",
                    "reason": str(exc),
                    "requested": str(requested),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    try:
        yield path, requested_digest
    finally:
        if run_root is not None:
            shutil.rmtree(run_root, ignore_errors=True)
            with suppress(OSError):
                run_root.parent.rmdir()


def build_receipt(
    *,
    entry: HostValidation,
    result: ShardResult,
    provenance: ExecutionProvenance,
    timing_digest: str,
    workers: int,
    total_worker_budget: int,
    max_parallel: int,
    store_durations: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "benchmark-host-validation-shard",
        **asdict(provenance),
        "timing_digest": timing_digest,
        "entry": entry.as_matrix_entry(),
        "workers": workers,
        "total_worker_budget": total_worker_budget,
        "max_parallel": max_parallel,
        "store_durations": store_durations,
        "command_digest": _command_digest(
            entry, workers=workers, store_durations=store_durations
        ),
        "status": result.status,
        "exit_code": result.exit_code,
        "actual_seconds": round(result.actual_seconds, 6),
    }
    payload["receipt_digest"] = receipt_digest(payload)
    return payload


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _default_runner(root: Path, environment: Mapping[str, str]) -> ShardRunner:
    def run(
        entry: HostValidation, arguments: tuple[str, ...], _workers: int
    ) -> ShardResult:
        from tools.pytest_lifecycle import run_pytest

        result = run_pytest(
            arguments,
            root=root,
            name=f"host-validation/{entry.name}",
            environment=environment,
        )
        return ShardResult(
            status=result.status,
            exit_code=result.exit_code,
            actual_seconds=result.actual_seconds,
        )

    return run


def execute(
    entries: Sequence[HostValidation],
    *,
    root: Path,
    timing_path: Path,
    receipt_dir: Path,
    provenance: ExecutionProvenance,
    total_worker_budget: int,
    max_parallel: int,
    store_durations: bool = False,
    duration_output: Path | None = None,
    runner: ShardRunner | None = None,
) -> int:
    """Execute selected entries under one explicit worker budget."""

    if not entries:
        raise ValueError("at least one host-validation entry is required")
    if store_durations and len(entries) != 1:
        raise ValueError("duration updates require one isolated shard process")
    if store_durations and duration_output is None:
        raise ValueError("duration updates require a separate output path")
    budgeted_slots = min(total_worker_budget, max_parallel)
    parallel, workers = worker_allocation(
        entry_count=len(entries),
        total_worker_budget=total_worker_budget,
        max_parallel=max_parallel,
    )
    run = runner or _default_runner(root, dict(os.environ))
    with timing_input(root, timing_path) as (resolved_timing, input_timing_digest):
        timing_run_root: Path | None = None
        execution_timing = resolved_timing
        if store_durations:
            timing_run_root = (
                root / ".pytest_cache" / "benchmark-host-timing" / uuid.uuid4().hex
            )
            timing_run_root.mkdir(parents=True)
            execution_timing = timing_run_root / "benchmark-test-durations.json"
            shutil.copyfile(resolved_timing, execution_timing)

        def run_one(entry: HostValidation) -> tuple[HostValidation, ShardResult]:
            print(
                json.dumps(
                    {
                        "event": "benchmark-host-shard-start",
                        "name": entry.name,
                        "workers": workers,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            result = run(
                entry,
                pytest_arguments(
                    entry,
                    timing_path=execution_timing,
                    workers=workers,
                    store_durations=store_durations,
                ),
                workers,
            )
            payload = build_receipt(
                entry=entry,
                result=result,
                provenance=provenance,
                timing_digest=input_timing_digest,
                workers=workers,
                total_worker_budget=total_worker_budget,
                max_parallel=budgeted_slots,
                store_durations=store_durations,
            )
            _write_receipt(receipt_dir / entry.name / "pytest-receipt.json", payload)
            print(
                json.dumps(
                    {
                        "event": "benchmark-host-shard-complete",
                        "name": entry.name,
                        "status": result.status,
                        "exit_code": result.exit_code,
                        "actual_seconds": round(result.actual_seconds, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return entry, result

        outcomes: list[tuple[HostValidation, ShardResult]] = []
        try:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [pool.submit(run_one, entry) for entry in entries]
                for future in as_completed(futures):
                    outcomes.append(future.result())
            if (
                store_durations
                and all(result.exit_code == 0 for _, result in outcomes)
                and duration_output is not None
            ):
                duration_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(execution_timing, duration_output)
        finally:
            if timing_run_root is not None:
                shutil.rmtree(timing_run_root, ignore_errors=True)
                with suppress(OSError):
                    timing_run_root.parent.rmdir()
    return 0 if all(result.exit_code == 0 for _, result in outcomes) else 1


def validate_receipts(
    root: Path,
    *,
    expected: Sequence[HostValidation],
    provenance: ExecutionProvenance,
    timing_digest: str,
    total_worker_budget: int = 8,
    max_parallel: int = 4,
) -> None:
    """Fail closed unless exactly one successful bound receipt exists per entry."""

    paths = sorted(root.rglob("pytest-receipt.json")) if root.is_dir() else []
    if len(paths) != len(expected):
        raise HarborSuiteError(
            f"expected {len(expected)} host receipts, found {len(paths)}"
        )
    expected_by_name = {entry.name: entry for entry in expected}
    seen: set[str] = set()
    for path in paths:
        entry = _validate_receipt(
            path,
            provenance=provenance,
            timing_digest=timing_digest,
            total_worker_budget=total_worker_budget,
            max_parallel=max_parallel,
        )
        if entry.name in seen or expected_by_name.get(entry.name) != entry:
            raise HarborSuiteError(
                f"unexpected or duplicate host-validation receipt: {entry.name}"
            )
        seen.add(entry.name)
    if seen != set(expected_by_name):
        raise HarborSuiteError(
            "host-validation receipts do not cover the planned matrix"
        )


def _validate_receipt(
    path: Path,
    *,
    provenance: ExecutionProvenance,
    timing_digest: str,
    total_worker_budget: int,
    max_parallel: int,
) -> HostValidation:
    payload = _read_json_object(path, "host-validation receipt")
    declared = _require_digest(payload.get("receipt_digest"), "shard receipt digest")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_digest"}
    if receipt_digest(unsigned) != declared:
        raise HarborSuiteError(f"host-validation receipt digest mismatch: {path}")
    if payload.get("schema_version") != 1 or payload.get("kind") != (
        "benchmark-host-validation-shard"
    ):
        raise HarborSuiteError(f"unsupported host-validation receipt: {path}")
    for field, value in asdict(provenance).items():
        if payload.get(field) != value:
            raise HarborSuiteError(f"host-validation receipt {field} mismatch: {path}")
    if payload.get("timing_digest") != timing_digest:
        raise HarborSuiteError(f"host-validation receipt timing mismatch: {path}")
    entry = _entry(payload.get("entry"))
    workers = payload.get("workers")
    total = payload.get("total_worker_budget")
    parallel = payload.get("max_parallel")
    store_durations = payload.get("store_durations")
    valid_budget = (
        isinstance(workers, int)
        and not isinstance(workers, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and isinstance(parallel, int)
        and not isinstance(parallel, bool)
        and workers >= 1
        and parallel >= 1
        and workers * parallel <= total
        and total == total_worker_budget
        and parallel == min(total_worker_budget, max_parallel)
        and store_durations is True
    )
    if not valid_budget:
        raise HarborSuiteError(f"host-validation receipt exceeds worker budget: {path}")
    if payload.get("command_digest") != _command_digest(
        entry,
        workers=cast(int, workers),
        store_durations=cast(bool, store_durations),
    ):
        raise HarborSuiteError(f"host-validation receipt command mismatch: {path}")
    actual_seconds = payload.get("actual_seconds")
    if (
        isinstance(actual_seconds, bool)
        or not isinstance(actual_seconds, (int, float))
        or not math.isfinite(float(actual_seconds))
        or actual_seconds < 0
    ):
        raise HarborSuiteError(f"host-validation receipt duration is invalid: {path}")
    if payload.get("status") != "EXITED" or payload.get("exit_code") != 0:
        raise HarborSuiteError(f"host-validation shard did not succeed: {entry.name}")
    return entry


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _resolve_plan(
    args: argparse.Namespace,
) -> tuple[ExecutionProvenance, tuple[HostValidation, ...]]:
    if args.plan is None:
        if args.execution_sha is not None:
            raise HarborSuiteError("--execution-sha requires --plan")
        planned = full_host_validation()
        return local_provenance(ROOT, planned), planned
    if args.execution_sha is None:
        raise HarborSuiteError("--execution-sha is required with --plan")
    execution_sha = verify_execution_sha(ROOT, args.execution_sha)
    return load_plan(args.plan, execution_sha=execution_sha)


def _selected_entries(
    args: argparse.Namespace, planned: tuple[HostValidation, ...]
) -> tuple[HostValidation, ...]:
    if args.command == "run-entry":
        entry = _entry(json.loads(args.entry_json))
        if entry not in planned:
            raise HarborSuiteError("requested host entry is absent from the plan")
        return (entry,)
    entries = full_host_validation()
    if args.plan is not None and tuple(planned) != entries:
        raise HarborSuiteError("run-full requires the complete full-host matrix")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run-full", "run-entry"):
        child = subparsers.add_parser(command)
        child.add_argument("--timings", type=Path, default=TIMING_PATH)
        child.add_argument("--receipt-dir", type=Path)
        child.add_argument("--total-workers", type=_positive_int, required=True)
        child.add_argument("--max-parallel", type=_positive_int, default=4)
        child.add_argument("--plan", type=Path)
        child.add_argument("--execution-sha")
        child.add_argument("--store-durations", action="store_true")
        child.add_argument("--durations-output", type=Path)
    entry_parser = subparsers.choices["run-entry"]
    entry_parser.add_argument("--entry-json", required=True)
    args = parser.parse_args(argv)
    try:
        provenance, planned = _resolve_plan(args)
        entries = _selected_entries(args, planned)
        local_receipt_root: Path | None = None
        receipt_dir = args.receipt_dir
        if receipt_dir is None:
            local_receipt_root = (
                ROOT / ".pytest_cache" / "benchmark-host-receipts" / uuid.uuid4().hex
            )
            receipt_dir = local_receipt_root
        try:
            return execute(
                entries,
                root=ROOT,
                timing_path=args.timings,
                receipt_dir=receipt_dir,
                provenance=provenance,
                total_worker_budget=args.total_workers,
                max_parallel=args.max_parallel,
                store_durations=args.store_durations,
                duration_output=args.durations_output,
            )
        finally:
            if local_receipt_root is not None:
                shutil.rmtree(local_receipt_root, ignore_errors=True)
    except (HarborSuiteError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExecutionProvenance",
    "ShardResult",
    "build_receipt",
    "execute",
    "load_plan",
    "local_provenance",
    "pytest_arguments",
    "timing_digest",
    "timing_input",
    "validate_receipts",
    "verify_execution_sha",
    "worker_allocation",
]
