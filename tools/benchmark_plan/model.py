"""Canonical Harbor/benchmark plan object and GitHub output projection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PLAN_VERSION = 1
EVENTS = ("pull_request", "merge_group", "push", "schedule", "workflow_dispatch")
MODES = ("none", "changed", "integration", "full")
SCOPES = ("none", "changed-tasks", "affected-datasets", "all")
PLAN_KEYS = (
    "schema_version",
    "event",
    "base_sha",
    "head_sha",
    "changed_paths_digest",
    "planner_digest",
    "topology_digest",
    "mode",
    "run_check",
    "record_schema",
    "prospective_digest",
    "inventory",
    "host_matrix",
    "oracle_scope",
    "oracle_matrix",
    "reasons",
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class BenchmarkPlan:
    """One content-bound Harbor plan. Validate with ``validate_plan``."""

    event: str
    base_sha: str
    head_sha: str
    changed_paths_digest: str
    planner_digest: str
    topology_digest: str
    mode: str
    run_check: bool
    record_schema: bool
    prospective_digest: bool
    inventory: bool
    host_matrix: tuple[dict[str, Any], ...]
    oracle_scope: str
    oracle_matrix: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]
    schema_version: int = PLAN_VERSION

    @property
    def run_host_validation(self) -> bool:
        return bool(self.host_matrix)

    @property
    def run_oracle(self) -> bool:
        return bool(self.oracle_matrix)

    @property
    def selected_checks(self) -> tuple[str, ...]:
        selected: list[str] = []
        if self.record_schema:
            selected.append("contracts")
        if self.run_host_validation:
            selected.append("host")
        if self.inventory:
            selected.append("inventory")
        if self.run_oracle:
            selected.append("oracle")
        if self.prospective_digest:
            selected.append("prospective_digest")
        return tuple(selected)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event": self.event,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "changed_paths_digest": self.changed_paths_digest,
            "planner_digest": self.planner_digest,
            "topology_digest": self.topology_digest,
            "mode": self.mode,
            "run_check": self.run_check,
            "record_schema": self.record_schema,
            "prospective_digest": self.prospective_digest,
            "inventory": self.inventory,
            "host_matrix": list(self.host_matrix),
            "oracle_scope": self.oracle_scope,
            "oracle_matrix": list(self.oracle_matrix),
            "reasons": list(self.reasons),
        }

    def github_outputs(self) -> dict[str, str]:
        """Project ``if:`` conditions and matrices for GitHub Actions."""

        return {
            "run-benchmark-check": str(self.run_check).lower(),
            "run-benchmark-record-schema": str(self.record_schema).lower(),
            "run-benchmark-inventory": str(self.inventory).lower(),
            "run-benchmark-host-validation": str(self.run_host_validation).lower(),
            "benchmark-host-validation-matrix": _json(list(self.host_matrix)),
            "run-benchmark-oracle": str(self.run_oracle).lower(),
            "benchmark-oracle-matrix": _json(list(self.oracle_matrix)),
        }


def plan_from_mapping(payload: Mapping[str, Any]) -> BenchmarkPlan:
    """Build a plan object from a JSON mapping without validating it."""

    host_matrix = payload.get("host_matrix", [])
    oracle_matrix = payload.get("oracle_matrix", [])
    reasons = payload.get("reasons", [])
    return BenchmarkPlan(
        schema_version=int(payload.get("schema_version", PLAN_VERSION)),
        event=str(payload.get("event", "")),
        base_sha=str(payload.get("base_sha", "")),
        head_sha=str(payload.get("head_sha", "")),
        changed_paths_digest=str(payload.get("changed_paths_digest", "")),
        planner_digest=str(payload.get("planner_digest", "")),
        topology_digest=str(payload.get("topology_digest", "")),
        mode=str(payload.get("mode", "")),
        run_check=bool(payload.get("run_check", False)),
        record_schema=bool(payload.get("record_schema", False)),
        prospective_digest=bool(payload.get("prospective_digest", False)),
        inventory=bool(payload.get("inventory", False)),
        host_matrix=tuple(host_matrix) if isinstance(host_matrix, Sequence) else (),
        oracle_scope=str(payload.get("oracle_scope", "none")),
        oracle_matrix=tuple(oracle_matrix)
        if isinstance(oracle_matrix, Sequence)
        else (),
        reasons=tuple(reasons) if isinstance(reasons, Sequence) else (),
    )


__all__ = [
    "EVENTS",
    "MODES",
    "PLAN_KEYS",
    "PLAN_VERSION",
    "SCOPES",
    "BenchmarkPlan",
    "plan_from_mapping",
]
