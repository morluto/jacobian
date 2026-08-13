"""Validate the benchmark planner's GitHub Actions output contract.

The planner emits a versioned plan bound to the triggering event and the
base/head (or merge-group) SHA, a digest of the planner, and a digest of the
benchmark topology. This validator enforces the full fail-closed contract:
every expected key is present, SHA bindings are well formed, the record/schema,
host-verifier, prospective-digest, inventory, and Oracle evidence roles are
mutually consistent, and an unselected role is never confused with a failed
one.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from typing import Any, NoReturn

from tools.benchmark_plan.compiler import EVENTS, MODES, PLAN_VERSION

SCOPES = {"none", "changed-tasks", "affected-datasets", "all"}
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SHA = re.compile(r"[0-9a-f]{40,64}\Z")
HOST_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
HOST_SELECTOR = re.compile(
    r"benchmarks/validation"
    r"(?:/(?![\./])(?:[A-Za-z0-9_][A-Za-z0-9_.-]*)"
    r"(?:/(?![\./])(?:[A-Za-z0-9_][A-Za-z0-9_.-]*))*)?\Z"
)
HOST_KEYWORD = re.compile(r"[A-Za-z0-9_.-]*\Z")
MAX_MATRIX_JOBS = 256

KEYS = {
    "benchmark-plan-version",
    "benchmark-plan-event",
    "benchmark-plan-base-sha",
    "benchmark-plan-head-sha",
    "benchmark-planner-digest",
    "benchmark-topology-digest",
    "benchmark-plan-mode",
    "run-benchmark-check",
    "run-benchmark-record-schema",
    "run-benchmark-prospective-digest",
    "run-benchmark-inventory",
    "run-benchmark-host-validation",
    "benchmark-host-validation-matrix",
    "run-benchmark-oracle",
    "benchmark-oracle-scope",
    "benchmark-oracle-matrix",
    "benchmark-plan-reasons",
}


class BenchmarkPlanValidationError(ValueError):
    """The emitted benchmark plan violates its fail-closed contract."""


def fail(message: str) -> NoReturn:
    raise BenchmarkPlanValidationError(message)


def _json_array(value: str, name: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        fail(f"{name} is not valid JSON: {exc}")
    if not isinstance(parsed, list):
        fail(f"{name} must be a JSON array")
    return parsed


def _bool(plan: dict[str, str], key: str) -> bool:
    value = plan[key]
    if value not in {"true", "false"}:
        fail(f"invalid benchmark boolean {key}: {value}")
    return value == "true"


def _require_sha(value: str, name: str) -> None:
    if value == "":
        return
    if SHA.fullmatch(value) is None:
        fail(f"invalid {name}: {value}")


def _require_positive_seconds(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        fail(label)


def _validate_task_digests(index: int, tasks: list[str], digests: Any) -> None:
    if not isinstance(digests, list) or len(digests) != len(tasks):
        fail(f"benchmark Oracle matrix entry {index} has invalid task digests")
    digest_tasks: set[str] = set()
    for item in digests:
        if not isinstance(item, dict) or set(item) != {"task", "digest"}:
            fail(f"benchmark Oracle matrix entry {index} has an invalid task digest")
        task = item["task"]
        digest = item["digest"]
        if not isinstance(task, str) or task not in tasks:
            fail(f"benchmark Oracle matrix entry {index} has an unknown digest task")
        if not isinstance(digest, str) or DIGEST.fullmatch(digest) is None:
            fail(f"benchmark Oracle matrix entry {index} has an invalid digest")
        if task in digest_tasks:
            fail(f"benchmark Oracle matrix entry {index} repeats a task digest")
        digest_tasks.add(task)
    if digest_tasks != set(tasks):
        fail(f"benchmark Oracle matrix entry {index} does not bind every task")


def _validate_oracle_entry(
    index: int,
    entry: Any,
    shard_pairs: set[tuple[str, str]],
    task_pairs: set[tuple[str, str]],
) -> None:
    if not isinstance(entry, dict) or set(entry) != {
        "dataset",
        "shard",
        "tasks",
        "task_digests",
        "predicted_seconds",
    }:
        fail(f"benchmark Oracle matrix entry {index} has an invalid shape")
    dataset = entry["dataset"]
    shard = entry["shard"]
    tasks = entry["tasks"]
    if not isinstance(dataset, str) or not dataset:
        fail(f"benchmark Oracle matrix entry {index} has an invalid dataset")
    if not isinstance(shard, str) or not shard:
        fail(f"benchmark Oracle matrix entry {index} has an invalid shard")
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(task, str) and task for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        fail(f"benchmark Oracle matrix entry {index} has invalid tasks")
    _require_positive_seconds(
        entry["predicted_seconds"],
        f"benchmark Oracle matrix entry {index} has invalid prediction",
    )
    _validate_task_digests(index, tasks, entry["task_digests"])

    shard_pair = (dataset, shard)
    if shard_pair in shard_pairs:
        fail(f"duplicate benchmark Oracle shard: {dataset}/{shard}")
    shard_pairs.add(shard_pair)
    for task in tasks:
        task_pair = (dataset, task)
        if task_pair in task_pairs:
            fail(f"duplicate benchmark Oracle matrix entry: {dataset}/{task}")
        task_pairs.add(task_pair)


def _validate_matrix(matrix: list[Any]) -> None:
    if len(matrix) > MAX_MATRIX_JOBS:
        fail(f"benchmark Oracle matrix exceeds {MAX_MATRIX_JOBS} jobs")
    shard_pairs: set[tuple[str, str]] = set()
    task_pairs: set[tuple[str, str]] = set()
    for index, entry in enumerate(matrix):
        _validate_oracle_entry(index, entry, shard_pairs, task_pairs)


def _validate_host_shard_fields(index: int, splits: Any, group: Any) -> tuple[int, int]:
    if (
        not isinstance(splits, int)
        or isinstance(splits, bool)
        or splits < 0
        or splits > MAX_MATRIX_JOBS
    ):
        fail(f"benchmark host validation matrix entry {index} has invalid splits")
    if not isinstance(group, int) or isinstance(group, bool):
        fail(f"benchmark host validation matrix entry {index} has invalid group")
    if splits == 0 and group != 0:
        fail(f"benchmark host validation matrix entry {index} has invalid group")
    if splits > 0 and not 1 <= group <= splits:
        fail(f"benchmark host validation matrix entry {index} has invalid group")
    return splits, group


def _validate_host_entry(
    index: int,
    entry: Any,
    names: set[str],
    shard_groups: dict[tuple[str, str], tuple[int, set[int]]],
) -> None:
    if not isinstance(entry, dict) or set(entry) != {
        "name",
        "selector",
        "keyword",
        "splits",
        "group",
        "predicted_seconds",
    }:
        fail(f"benchmark host validation matrix entry {index} has an invalid shape")
    name = entry["name"]
    selector = entry["selector"]
    keyword = entry["keyword"]
    if (
        not isinstance(name, str)
        or name in {".", ".."}
        or HOST_NAME.fullmatch(name) is None
    ):
        fail(f"benchmark host validation matrix entry {index} has an invalid name")
    if name in names:
        fail(f"duplicate benchmark host validation matrix name: {name}")
    names.add(name)
    if (
        not isinstance(selector, str)
        or HOST_SELECTOR.fullmatch(selector) is None
        or ".." in selector.split("/")
    ):
        fail(f"benchmark host validation matrix entry {index} has an invalid selector")
    if not isinstance(keyword, str) or HOST_KEYWORD.fullmatch(keyword) is None:
        fail(f"benchmark host validation matrix entry {index} has an invalid keyword")
    splits, group = _validate_host_shard_fields(index, entry["splits"], entry["group"])
    _require_positive_seconds(
        entry["predicted_seconds"],
        f"benchmark host validation matrix entry {index} has invalid prediction",
    )
    key = (selector, keyword)
    expected_splits, groups = shard_groups.setdefault(key, (splits, set()))
    if expected_splits != splits:
        fail(f"inconsistent benchmark host validation sharding: {selector}")
    if group in groups:
        fail(f"duplicate benchmark host validation matrix entry: {selector}")
    groups.add(group)


def _validate_host_matrix(matrix: list[Any]) -> None:
    if len(matrix) > MAX_MATRIX_JOBS:
        fail(f"benchmark host validation matrix exceeds {MAX_MATRIX_JOBS} jobs")
    names: set[str] = set()
    shard_groups: dict[tuple[str, str], tuple[int, set[int]]] = {}
    for index, entry in enumerate(matrix):
        _validate_host_entry(index, entry, names, shard_groups)
    for (selector, _keyword), (splits, groups) in shard_groups.items():
        expected_groups = {0} if splits == 0 else set(range(1, splits + 1))
        if groups != expected_groups:
            fail(f"incomplete benchmark host validation sharding: {selector}")


def _host_validation_selection(plan: dict[str, str], *, run_check: bool) -> bool:
    selected = _bool(plan, "run-benchmark-host-validation")
    matrix = _json_array(
        plan["benchmark-host-validation-matrix"],
        "benchmark host validation matrix",
    )
    if selected and not run_check:
        fail("benchmark host validation requires benchmark checks")
    if selected != bool(matrix):
        fail("benchmark host validation flag and matrix disagree")
    _validate_host_matrix(matrix)
    return selected


def _validate_identity(plan: dict[str, str]) -> str:
    if set(plan) != KEYS:
        fail(f"benchmark plan keys differ: expected {sorted(KEYS)}, got {sorted(plan)}")
    if plan["benchmark-plan-version"] != PLAN_VERSION:
        fail(f"unsupported benchmark plan version: {plan['benchmark-plan-version']}")
    if plan["benchmark-plan-event"] not in EVENTS:
        fail(f"invalid benchmark plan event: {plan['benchmark-plan-event']}")
    mode = plan["benchmark-plan-mode"]
    if mode not in MODES:
        fail(f"invalid benchmark plan mode: {mode}")
    _require_sha(plan["benchmark-plan-base-sha"], "benchmark-plan-base-sha")
    _require_sha(plan["benchmark-plan-head-sha"], "benchmark-plan-head-sha")
    if DIGEST.fullmatch(plan["benchmark-planner-digest"]) is None:
        fail(f"invalid benchmark-planner-digest: {plan['benchmark-planner-digest']}")
    return mode


def _validate_check_topology(
    plan: dict[str, str],
    *,
    run_check: bool,
    mode: str,
    record_schema: bool,
    reasons: list[Any],
) -> None:
    topology = plan["benchmark-topology-digest"]
    if run_check:
        if DIGEST.fullmatch(topology) is None:
            fail("benchmark-topology-digest must be a sha256 digest when checks run")
        if mode == "none":
            fail("benchmark plan mode must not be none when checks run")
        if not record_schema:
            fail("run-benchmark-record-schema must run when benchmark checks run")
        if not reasons:
            fail("a benchmark plan with work must record reasons")
    elif topology != "":
        fail("benchmark-topology-digest must be empty when checks are skipped")


def _validate_lane_independence(
    *,
    run_check: bool,
    mode: str,
    prospective_digest: bool,
    inventory: bool,
    record_schema: bool,
    host_validation: bool,
    run_oracle: bool,
    reasons: list[Any],
) -> None:
    if prospective_digest and not run_check:
        fail("run-benchmark-prospective-digest requires benchmark checks")
    if inventory and not run_check:
        fail("run-benchmark-inventory requires benchmark checks")
    if inventory and mode not in {"integration", "full"}:
        fail("run-benchmark-inventory requires integration or full mode")
    if mode in {"integration", "full"} and not run_check:
        fail("integration or full mode requires benchmark checks")
    if mode == "none" and run_check:
        fail("benchmark plan mode none conflicts with running checks")
    if not run_check and (
        record_schema
        or prospective_digest
        or inventory
        or host_validation
        or run_oracle
        or reasons
    ):
        fail("a skipped benchmark plan cannot contain work or reasons")


def _validate_oracle_selection(
    plan: dict[str, str],
    *,
    run_check: bool,
    record_schema: bool,
) -> None:
    run_oracle = _bool(plan, "run-benchmark-oracle")
    scope = plan["benchmark-oracle-scope"]
    if scope not in SCOPES:
        fail(f"invalid benchmark Oracle scope: {scope}")
    matrix = _json_array(plan["benchmark-oracle-matrix"], "benchmark Oracle matrix")
    if run_oracle:
        if not (run_check and record_schema):
            fail("benchmark Oracle work requires checks and record/schema lanes")
        if scope == "none" or not matrix:
            fail("benchmark Oracle work requires a scope and a non-empty matrix")
    elif scope != "none" or matrix:
        fail("disabled benchmark Oracle work must have none scope and empty matrix")
    _validate_matrix(matrix)


def validate_plan(plan: dict[str, str]) -> None:
    """Validate a complete benchmark plan mapping."""

    mode = _validate_identity(plan)
    run_check = _bool(plan, "run-benchmark-check")
    record_schema = _bool(plan, "run-benchmark-record-schema")
    prospective_digest = _bool(plan, "run-benchmark-prospective-digest")
    inventory = _bool(plan, "run-benchmark-inventory")
    run_oracle = _bool(plan, "run-benchmark-oracle")
    host_validation = _host_validation_selection(plan, run_check=run_check)
    reasons = _json_array(plan["benchmark-plan-reasons"], "benchmark plan reasons")
    if not all(isinstance(reason, str) and reason for reason in reasons):
        fail("benchmark plan reasons must be non-empty strings")
    if prospective_digest and not record_schema:
        fail("run-benchmark-prospective-digest requires record/schema checks")
    _validate_check_topology(
        plan,
        run_check=run_check,
        mode=mode,
        record_schema=record_schema,
        reasons=reasons,
    )
    _validate_lane_independence(
        run_check=run_check,
        mode=mode,
        prospective_digest=prospective_digest,
        inventory=inventory,
        record_schema=record_schema,
        host_validation=host_validation,
        run_oracle=run_oracle,
        reasons=reasons,
    )
    _validate_oracle_selection(plan, run_check=run_check, record_schema=record_schema)


def _read_plan(lines: Iterable[str]) -> dict[str, str]:
    plan: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if "=" not in line:
            fail(f"invalid benchmark plan line: {line}")
        key, value = line.split("=", 1)
        if key in plan:
            fail(f"duplicate benchmark plan key: {key}")
        plan[key] = value
    return plan


def main() -> int:
    """Read a key=value plan from stdin and return a CLI status."""

    try:
        validate_plan(_read_plan(sys.stdin))
    except BenchmarkPlanValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


__all__ = [
    "EVENTS",
    "MODES",
    "PLAN_VERSION",
    "BenchmarkPlanValidationError",
    "main",
    "validate_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
