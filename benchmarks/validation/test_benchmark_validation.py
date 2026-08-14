"""Fail-closed tests for the stable benchmark result aggregator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from benchmarks.tooling.benchmark_validation import LaneResult, validate_aggregate
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.host_validation import (
    ExecutionProvenance,
    ShardResult,
    build_receipt,
    load_plan,
)
from benchmarks.tooling.receipts import digest_bytes, receipt_digest
from benchmarks.tooling.validation_plan import HostValidation, full_host_validation

DIGEST = "sha256:" + "a" * 64
SHA = "1" * 40
EXECUTION_SHA = "2" * 40


def _digest(value: object) -> str:
    return receipt_digest(value)


def _plan_payload(
    entries: tuple[HostValidation, ...],
    *,
    selected: bool = True,
) -> dict[str, Any]:
    if selected:
        return {
            "schema_version": 1,
            "event": "pull_request",
            "base_sha": SHA,
            "head_sha": SHA,
            "changed_paths_digest": DIGEST,
            "planner_digest": DIGEST,
            "topology_digest": DIGEST,
            "mode": "changed",
            "run_check": True,
            "record_schema": True,
            "prospective_digest": False,
            "inventory": False,
            "host_matrix": [entry.as_matrix_entry() for entry in entries],
            "oracle_scope": "none",
            "oracle_matrix": [],
            "reasons": ["host validation"],
        }
    return {
        "schema_version": 1,
        "event": "pull_request",
        "base_sha": SHA,
        "head_sha": SHA,
        "changed_paths_digest": "",
        "planner_digest": DIGEST,
        "topology_digest": "",
        "mode": "none",
        "run_check": False,
        "record_schema": False,
        "prospective_digest": False,
        "inventory": False,
        "host_matrix": [],
        "oracle_scope": "none",
        "oracle_matrix": [],
        "reasons": [],
    }


def _write_plan(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _plan_file(tmp_path: Path) -> tuple[Path, ExecutionProvenance]:
    entries = full_host_validation()
    path = _write_plan(tmp_path, _plan_payload(entries))
    provenance, observed = load_plan(path, execution_sha=EXECUTION_SHA)
    assert observed == entries
    return path, provenance


def _evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    plan_path, provenance = _plan_file(tmp_path)
    timings = tmp_path / "benchmark-test-durations.json"
    timings.write_text("{}\n", encoding="utf-8")
    timing_digest = digest_bytes(b"{}\n")
    receipt_root = tmp_path / "receipts"
    for entry in full_host_validation():
        payload = build_receipt(
            entry=entry,
            result=ShardResult(status="EXITED", exit_code=0, actual_seconds=3.0),
            provenance=provenance,
            timing_digest=timing_digest,
            workers=2,
            total_worker_budget=8,
            max_parallel=4,
            store_durations=True,
        )
        path = receipt_root / entry.name / "pytest-receipt.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return plan_path, receipt_root, timings


def _lanes() -> dict[str, LaneResult]:
    return {
        "static": LaneResult(True, "success"),
        "contracts": LaneResult(True, "success"),
        "host-validation": LaneResult(True, "success"),
        "inventory": LaneResult(False, "skipped"),
        "oracle": LaneResult(False, "skipped"),
    }


def test_aggregate_accepts_exact_successful_shard_receipts(tmp_path: Path) -> None:
    plan, receipts, timings = _evidence(tmp_path)

    validate_aggregate(
        plan_result="success",
        plan_path=plan,
        execution_sha=EXECUTION_SHA,
        lanes=_lanes(),
        receipt_root=receipts,
        timing_path=timings,
    )


def test_canonical_plan_round_trips_through_consumer(tmp_path: Path) -> None:
    entries = full_host_validation()
    path = _write_plan(tmp_path, _plan_payload(entries))

    provenance, observed = load_plan(path, execution_sha=EXECUTION_SHA)

    assert observed == entries
    assert provenance.plan_head_sha == SHA
    assert provenance.planner_digest == DIGEST


def test_aggregate_accepts_consistent_empty_plan(tmp_path: Path) -> None:
    path = _write_plan(tmp_path, _plan_payload((), selected=False))
    lanes = {name: LaneResult(False, "skipped") for name in _lanes()}

    validate_aggregate(
        plan_result="success",
        plan_path=path,
        execution_sha=EXECUTION_SHA,
        lanes=lanes,
        receipt_root=None,
        timing_path=None,
    )


@pytest.mark.parametrize("failure", ["missing", "stale", "failed"])
def test_aggregate_rejects_invalid_host_evidence(tmp_path: Path, failure: str) -> None:
    plan, receipts, timings = _evidence(tmp_path)
    first = next(receipts.rglob("pytest-receipt.json"))
    if failure == "missing":
        first.unlink()
    else:
        payload = json.loads(first.read_text(encoding="utf-8"))
        if failure == "stale":
            payload["execution_sha"] = "3" * 40
        else:
            payload["exit_code"] = 1
        unsigned = {
            key: value for key, value in payload.items() if key != "receipt_digest"
        }
        payload["receipt_digest"] = _digest(unsigned)
        first.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HarborSuiteError):
        validate_aggregate(
            plan_result="success",
            plan_path=plan,
            execution_sha=EXECUTION_SHA,
            lanes=_lanes(),
            receipt_root=receipts,
            timing_path=timings,
        )


def test_aggregate_rejects_selected_lane_that_was_skipped(tmp_path: Path) -> None:
    plan, receipts, timings = _evidence(tmp_path)
    lanes = _lanes()
    lanes["static"] = LaneResult(True, "skipped")

    with pytest.raises(HarborSuiteError, match="static expected success"):
        validate_aggregate(
            plan_result="success",
            plan_path=plan,
            execution_sha=EXECUTION_SHA,
            lanes=lanes,
            receipt_root=receipts,
            timing_path=timings,
        )
