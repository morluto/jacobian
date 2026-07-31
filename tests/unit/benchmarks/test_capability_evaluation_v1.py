from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parents[3]
EVALUATION = ROOT / "benchmarks" / "capability-evaluations" / "v1"
REGRESSION = ROOT / "benchmarks" / "regression-v1"


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
        path.name for path in (REGRESSION / "tasks").iterdir() if path.is_dir()
    )

    assert ledger_task_ids == task_dirs
    assert len(ledger_task_ids) == len(set(ledger_task_ids)) == 24
    manifest_bytes = (REGRESSION / "dataset.toml").read_bytes()
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


def test_public_c1_and_c2_jobs_differ_only_by_condition_identity() -> None:
    c1 = _load("job-public-c1-current.json")
    c2 = _load("job-public-c2-treatment.json")

    assert c1["agents"][0]["model_name"] == "${JACOBIAN_MODEL}"
    assert c2["agents"][0]["model_name"] == "${JACOBIAN_MODEL}"
    assert (
        c1["datasets"] == c2["datasets"] == [{"path": "benchmarks/regression-v1/tasks"}]
    )

    normalized_c1 = deepcopy(c1)
    normalized_c2 = deepcopy(c2)
    normalized_c1["jobs_dir"] = normalized_c2["jobs_dir"] = "<condition-results>"
    normalized_c1["environment"]["extra_docker_compose"] = ["<condition-compose>"]
    normalized_c2["environment"]["extra_docker_compose"] = ["<condition-compose>"]
    assert normalized_c1 == normalized_c2


def test_public_condition_compose_files_differ_only_by_image_selection() -> None:
    c1 = (EVALUATION / "public-c1-current.compose.yaml").read_text(encoding="utf-8")
    c2 = (EVALUATION / "public-c2-treatment.compose.yaml").read_text(encoding="utf-8")

    assert "JACOBIAN_C1_IMAGE must be digest-pinned" in c1
    assert "JACOBIAN_C2_IMAGE must be digest-pinned" in c2
    assert c1.replace("JACOBIAN_C1_IMAGE", "JACOBIAN_CONDITION_IMAGE") == c2.replace(
        "JACOBIAN_C2_IMAGE", "JACOBIAN_CONDITION_IMAGE"
    )
    assert not (EVALUATION / "held-out").exists()
