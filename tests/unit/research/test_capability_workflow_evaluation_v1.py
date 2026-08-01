from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parents[3]
EVALUATION = ROOT / "research" / "evaluations" / "capability-workflow-v1"
WORKFLOW = ROOT / "benchmarks" / "datasets" / "agent-workflow-v1"


def _load(name: str) -> dict[str, object]:
    payload = json.loads((EVALUATION / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _validate(instance_name: str, schema_name: str) -> dict[str, object]:
    instance = _load(instance_name)
    schema = _load(schema_name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)
    return instance


def test_gap_ledger_covers_the_frozen_public_suite() -> None:
    ledger = _validate("gap-ledger.json", "gap-ledger.schema.json")
    tasks = ledger["tasks"]
    assert isinstance(tasks, list)
    ledger_task_ids = [entry["task_id"] for entry in tasks]
    task_dirs = sorted(
        path.parent.name for path in (WORKFLOW / "tasks").rglob("task.toml")
    )

    assert ledger_task_ids == task_dirs
    assert len(ledger_task_ids) == len(set(ledger_task_ids)) == 26
    manifest_bytes = (WORKFLOW / "dataset.toml").read_bytes()
    manifest_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    assert ledger["source_suite"]["manifest_sha256"] == manifest_digest


def test_gap_ledger_handoffs_are_closed_over_task_candidates() -> None:
    ledger = _load("gap-ledger.json")
    tasks = ledger["tasks"]
    handoffs = ledger["handoffs"]
    assert isinstance(tasks, list)
    assert isinstance(handoffs, list)
    handoff_by_id = {entry["subject"]["candidate_id"]: entry for entry in handoffs}
    task_candidates = {
        candidate_id for task in tasks for candidate_id in task["candidate_ids"]
    }

    workflow_task_refs = {
        path.parent.relative_to(ROOT).as_posix()
        for path in (WORKFLOW / "tasks").rglob("task.toml")
    }
    for handoff in handoffs:
        for evidence in handoff.get("evidence_refs", []):
            ref = evidence["ref"]
            if ref.startswith("benchmarks/datasets/agent-workflow-v1/"):
                assert ref in workflow_task_refs
                assert (ROOT / ref / "task.toml").is_file()

    assert task_candidates == set(handoff_by_id)
    for candidate_id, handoff in handoff_by_id.items():
        if handoff["status"] == "accepted":
            assert handoff["decision"] == "IMPLEMENT"
            assert handoff["subject"]["capability_ids"]
            assert handoff["candidate_gate"]["basis"] in {
                "RECURRENT",
                "FUNDAMENTAL_PRIMITIVE",
            }
        else:
            assert handoff["decision"] == "DEFER"
            assert handoff["missing_evidence"]
        assert candidate_id not in {
            "calendar-good-days-solver",
            "proof-audit",
            "distinct-sum-pairing-solver",
        }


def test_comparison_plan_is_fail_closed_until_treatment_and_held_out_freeze() -> None:
    plan = _validate("comparison-plan.json", "comparison-plan.schema.json")
    conditions = plan["conditions"]
    assert isinstance(conditions, list)
    assert [condition["condition_id"] for condition in conditions] == [
        "C0",
        "C1",
        "C2",
    ]
    assert conditions[1]["role"] == "PRIMARY_CONTROL"
    assert conditions[2]["role"] == "PRIMARY_TREATMENT"
    assert all(condition["public_job_config"] is None for condition in conditions)
    assert all(
        condition["image_environment_variable"] is None for condition in conditions
    )
    assert conditions[2]["catalog_digest"] is None
    assert plan["held_out_boundary"]["status"] == "NOT_MATERIALIZED"
    assert plan["execution_gate"]["model_execution_allowed"] is False
    assert plan["claim_policy"]["public_jobs_causal_claims_allowed"] is False

    ledger = _load("gap-ledger.json")
    assert (
        conditions[1]["catalog_digest"] == ledger["runtime_snapshot"]["catalog_digest"]
    )
    assert (
        plan["public_reproduction"]["manifest_digest"]
        == ledger["source_suite"]["manifest_sha256"]
    )


def test_research_scaffold_has_no_executable_harbor_inputs() -> None:
    assert sorted(path.name for path in EVALUATION.iterdir()) == [
        "README.md",
        "comparison-plan.json",
        "comparison-plan.schema.json",
        "gap-ledger.json",
        "gap-ledger.schema.json",
    ]
