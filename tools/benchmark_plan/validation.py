"""Validate the canonical Harbor benchmark plan object."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from typing import Any, NoReturn

from tools.benchmark_plan.model import (
    EVENTS,
    MODES,
    PLAN_KEYS,
    PLAN_VERSION,
    SCOPES,
    BenchmarkPlan,
)

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


class BenchmarkPlanValidationError(ValueError):
    """The emitted benchmark plan violates its fail-closed contract."""


def fail(message: str) -> NoReturn:
    raise BenchmarkPlanValidationError(message)


def _require_sha(value: str, name: str) -> None:
    if value == "":
        return
    if SHA.fullmatch(value) is None:
        fail(f"invalid {name}: {value}")


def _require_positive_seconds(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        fail(label)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        fail(f"invalid benchmark boolean {name}: {value}")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{name} must be a JSON array")
    return value


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


def _payload(plan: BenchmarkPlan | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, BenchmarkPlan):
        return plan.to_json_dict()
    if not isinstance(plan, Mapping):
        fail("benchmark plan must be an object")
    return dict(plan)


def _validate_identities(payload: Mapping[str, Any]) -> None:
    if set(payload) != set(PLAN_KEYS):
        fail(
            f"benchmark plan keys differ: expected {sorted(PLAN_KEYS)}, "
            f"got {sorted(payload)}"
        )
    if payload["schema_version"] != PLAN_VERSION:
        fail(f"unsupported benchmark plan version: {payload['schema_version']}")
    if payload["event"] not in EVENTS:
        fail(f"invalid benchmark plan event: {payload['event']}")
    if payload["mode"] not in MODES:
        fail(f"invalid benchmark plan mode: {payload['mode']}")
    _require_sha(str(payload["base_sha"]), "base-sha")
    _require_sha(str(payload["head_sha"]), "head-sha")
    planner_digest = payload["planner_digest"]
    if not isinstance(planner_digest, str) or DIGEST.fullmatch(planner_digest) is None:
        fail(f"invalid planner-digest: {planner_digest}")
    changed = payload["changed_paths_digest"]
    if changed != "" and (
        not isinstance(changed, str) or DIGEST.fullmatch(changed) is None
    ):
        fail(f"invalid changed_paths_digest: {changed}")


def _validate_check_topology(
    *,
    run_check: bool,
    record_schema: bool,
    mode: str,
    topology: object,
    reasons: list[Any],
) -> None:
    if run_check:
        if not isinstance(topology, str) or DIGEST.fullmatch(topology) is None:
            fail("topology-digest must be a sha256 digest when checks run")
        if mode == "none":
            fail("benchmark plan mode must not be none when checks run")
        if not record_schema:
            fail("record-schema must run when benchmark checks run")
        if not reasons:
            fail("a benchmark plan with work must record reasons")
        return
    if topology != "":
        fail("topology-digest must be empty when checks are skipped")


def _validate_inventory_and_mode(
    *,
    run_check: bool,
    prospective_digest: bool,
    inventory: bool,
    record_schema: bool,
    mode: str,
) -> None:
    if prospective_digest and not run_check:
        fail("prospective-digest requires benchmark checks")
    if inventory and not run_check:
        fail("inventory requires benchmark checks")
    if inventory and mode not in {"integration", "full"}:
        fail("inventory requires integration or full mode")
    if mode in {"integration", "full"} and not run_check:
        fail("integration or full mode requires benchmark checks")
    if mode == "none" and run_check:
        fail("benchmark plan mode none conflicts with running checks")
    if prospective_digest and not record_schema:
        fail("prospective-digest requires record/schema checks")


def _validate_oracle_scope(
    *,
    run_check: bool,
    record_schema: bool,
    run_oracle: bool,
    scope: object,
    oracle_matrix: list[Any],
) -> None:
    if scope not in SCOPES:
        fail(f"invalid benchmark Oracle scope: {scope}")
    if run_oracle:
        if not (run_check and record_schema):
            fail("benchmark Oracle work requires checks and record/schema lanes")
        if scope == "none":
            fail("benchmark Oracle work requires a scope and a non-empty matrix")
        return
    if scope != "none" or oracle_matrix:
        fail("disabled benchmark Oracle work must have none scope and empty matrix")


def _validate_lane_consistency(
    *,
    run_check: bool,
    record_schema: bool,
    prospective_digest: bool,
    inventory: bool,
    host_validation: bool,
    run_oracle: bool,
    mode: str,
    scope: object,
    topology: object,
    reasons: list[Any],
    oracle_matrix: list[Any],
) -> None:
    if not all(isinstance(reason, str) and reason for reason in reasons):
        fail("benchmark plan reasons must be non-empty strings")
    if host_validation and not run_check:
        fail("benchmark host validation requires benchmark checks")
    _validate_check_topology(
        run_check=run_check,
        record_schema=record_schema,
        mode=mode,
        topology=topology,
        reasons=reasons,
    )
    _validate_inventory_and_mode(
        run_check=run_check,
        prospective_digest=prospective_digest,
        inventory=inventory,
        record_schema=record_schema,
        mode=mode,
    )
    if not run_check and (
        record_schema
        or prospective_digest
        or inventory
        or host_validation
        or run_oracle
        or reasons
    ):
        fail("a skipped benchmark plan cannot contain work or reasons")
    _validate_oracle_scope(
        run_check=run_check,
        record_schema=record_schema,
        run_oracle=run_oracle,
        scope=scope,
        oracle_matrix=oracle_matrix,
    )


def validate_plan(plan: BenchmarkPlan | Mapping[str, Any]) -> None:
    """Validate a complete canonical benchmark plan."""

    payload = _payload(plan)
    _validate_identities(payload)
    run_check = _require_bool(payload["run_check"], "run_check")
    record_schema = _require_bool(payload["record_schema"], "record_schema")
    prospective_digest = _require_bool(
        payload["prospective_digest"], "prospective_digest"
    )
    inventory = _require_bool(payload["inventory"], "inventory")
    host_matrix = _require_list(payload["host_matrix"], "host_matrix")
    oracle_matrix = _require_list(payload["oracle_matrix"], "oracle_matrix")
    reasons = _require_list(payload["reasons"], "reasons")
    _validate_host_matrix(host_matrix)
    _validate_lane_consistency(
        run_check=run_check,
        record_schema=record_schema,
        prospective_digest=prospective_digest,
        inventory=inventory,
        host_validation=bool(host_matrix),
        run_oracle=bool(oracle_matrix),
        mode=str(payload["mode"]),
        scope=payload["oracle_scope"],
        topology=payload["topology_digest"],
        reasons=reasons,
        oracle_matrix=oracle_matrix,
    )
    _validate_matrix(oracle_matrix)


def main() -> int:
    """Read a JSON plan from stdin and return a CLI status."""

    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            fail("benchmark plan must be an object")
        validate_plan(payload)
    except (BenchmarkPlanValidationError, json.JSONDecodeError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


__all__ = [
    "EVENTS",
    "MAX_MATRIX_JOBS",
    "MODES",
    "PLAN_VERSION",
    "BenchmarkPlanValidationError",
    "main",
    "validate_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())
