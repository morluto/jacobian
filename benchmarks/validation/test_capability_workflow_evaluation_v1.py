from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.tooling.harbor_suite import get_suite
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parents[2]
EVALUATION = ROOT / "research" / "evaluations" / "capability-workflow-v1"
DATASET = ROOT / "benchmarks" / "datasets" / "agent-workflow-v1"
LEDGER_ID = "jacobian.agent-workflow-v1.capability-gap-ledger"


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


def _canonical_digest(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _leaf_schema(ledger_schema: dict[str, Any]) -> dict[str, Any]:
    """A standalone schema that validates one leaf against analysisLeaf."""
    return {
        "$schema": ledger_schema["$schema"],
        "$defs": ledger_schema["$defs"],
        "$ref": "#/$defs/analysisLeaf",
    }


def _load_leaf(record: dict[str, Any]) -> dict[str, Any]:
    leaf = json.loads((ROOT / str(record["analysis_ref"])).read_text(encoding="utf-8"))
    assert isinstance(leaf, dict)
    return leaf


def test_gap_ledger_validates_and_frozen_analysis_records_match_leaves() -> None:
    ledger = _validate("gap-ledger.json", "gap-ledger.schema.json")
    records = ledger["analysis_records"]
    assert isinstance(records, list)
    assert len(records) >= 1

    ledger_schema = _load("gap-ledger.schema.json")
    leaf_validator = Draft202012Validator(
        _leaf_schema(ledger_schema), format_checker=FormatChecker()
    )

    task_ids: list[str] = []
    refs: list[str] = []
    for index, record in enumerate(records):
        assert record["order"] == index
        assert record["order"] == list(range(len(records)))[index]
        task_ids.append(record["task_id"])
        refs.append(record["analysis_ref"])

        leaf = _load_leaf(record)
        leaf_validator.validate(leaf)
        assert leaf["ledger_id"] == LEDGER_ID
        assert leaf["task_id"] == record["task_id"]
        # Leaves are snapshot-independent: embedding snapshot_id would create a
        # digest cycle because the leaf is inside the Harbor task checksum.
        assert "snapshot" not in leaf
        # The frozen digest binds the leaf's analysis content, independent of
        # file formatting or surrounding metadata.
        assert _canonical_digest(leaf["analysis"]) == record["analysis_digest"]
        # The leaf lives inside its matching task bundle.
        assert record["analysis_ref"].startswith(
            f"benchmarks/datasets/agent-workflow-v1/{record['task_id']}/analysis/gap.json"
        )

    assert len(task_ids) == len(set(task_ids))
    assert len(refs) == len(set(refs))
    assert [record["order"] for record in records] == list(range(len(records)))


def test_historical_ledger_is_decoupled_from_current_suite_membership() -> None:
    ledger = _validate("gap-ledger.json", "gap-ledger.schema.json")
    # The historical ledger must not carry the old mutable coupling fields.
    assert "source_suite" not in ledger
    assert "tasks" not in ledger
    assert "manifest_sha256" not in ledger
    assert "manifest_digest" not in ledger
    # No hard-coded current task count: the frozen size is whatever was captured.
    assert "task_count" not in ledger
    snapshot = ledger["snapshot"]
    assert snapshot["status"] == "LOCKED"
    assert snapshot["snapshot_id"].startswith("sha256:")
    assert (ROOT / snapshot["lock_path"]).is_file()
    # The frozen records are inline data, not a live enumeration of the suite.
    assert isinstance(ledger["analysis_records"], list)


def test_current_task_additions_do_not_retarget_historical_records() -> None:
    ledger = _load("gap-ledger.json")
    records = ledger["analysis_records"]
    frozen_ids = [record["task_id"] for record in records]
    frozen_set = set(frozen_ids)

    current_ids = sorted(ref.path.name for ref in get_suite("agent-workflow-v1").tasks)
    current_set = set(current_ids)

    # Every historical task bundle still exists, so the leaf records remain
    # readable; the historical capture is a subset of the live suite.
    assert frozen_set <= current_set

    # A task added to the live suite after the historical capture must not be
    # pulled into the frozen records. This is the retargeting proof: the live
    # suite has grown, but the historical ledger has not followed it.
    post_capture = current_set - frozen_set
    assert post_capture, (
        "expected at least one post-capture task in the live suite to prove the "
        "historical ledger is not retargeted by current task additions"
    )
    for added in post_capture:
        assert added not in frozen_set
        # The ledger references no leaf for the added task.
        assert all(record["task_id"] != added for record in records)

    # The frozen records are fully determined by their leaf content digests and
    # order, with no dependency on the live suite enumeration. Recompute the
    # record identities purely from the leaves and confirm they match.
    rebuilt = [
        {
            "order": index,
            "task_id": record["task_id"],
            "analysis_ref": record["analysis_ref"],
            "analysis_digest": _canonical_digest(_load_leaf(record)["analysis"]),
        }
        for index, record in enumerate(records)
    ]
    assert rebuilt == [
        {
            "order": record["order"],
            "task_id": record["task_id"],
            "analysis_ref": record["analysis_ref"],
            "analysis_digest": record["analysis_digest"],
        }
        for record in records
    ]


def test_gap_ledger_handoffs_are_closed_over_task_candidates() -> None:
    ledger = _load("gap-ledger.json")
    records = ledger["analysis_records"]
    handoffs = ledger["handoffs"]
    assert isinstance(records, list)
    assert isinstance(handoffs, list)

    task_candidates = {
        candidate_id
        for record in records
        for candidate_id in _load_leaf(record)["analysis"]["candidate_ids"]
    }
    handoff_by_id = {
        handoff["subject"]["candidate_id"]: handoff for handoff in handoffs
    }

    # Handoff evidence refs that point at task bundles must resolve to real
    # Harbor task bundles on disk (the historical tasks still exist).
    for handoff in handoffs:
        for evidence in handoff.get("evidence_refs", []):
            ref = evidence["ref"]
            if ref.startswith("benchmarks/datasets/agent-workflow-v1/"):
                bundle = ROOT / ref
                assert (bundle / "task.toml").is_file(), f"missing bundle: {ref}"

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


def test_historical_oracle_coverage_is_distinguished_from_snapshot_task_set() -> None:
    ledger = _validate("gap-ledger.json", "gap-ledger.schema.json")
    oracle = ledger["runtime_snapshot"]["oracle_evidence"]
    record_count = len(ledger["analysis_records"])
    # Historical Oracle coverage is a captured subset, not the full frozen task
    # set and not the snapshot identity.
    assert isinstance(oracle["task_count"], int)
    assert oracle["task_count"] < record_count
    assert oracle["task_count"] != record_count
    # The immutable snapshot identity is distinct from the Oracle evidence.
    assert ledger["snapshot"]["status"] == "LOCKED"
    assert ledger["snapshot"]["snapshot_id"] != oracle.get("result_digest")


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
    # The plan shares the ledger's snapshot reference (one immutable snapshot),
    # not a mutable manifest digest or current task count.
    assert plan["public_reproduction"]["snapshot"] == ledger["snapshot"]
    assert "manifest_digest" not in plan["public_reproduction"]
    assert "task_count" not in plan["public_reproduction"]
    assert (
        conditions[1]["catalog_digest"] == ledger["runtime_snapshot"]["catalog_digest"]
    )


def test_research_scaffold_has_no_executable_harbor_inputs() -> None:
    assert sorted(path.name for path in EVALUATION.iterdir()) == [
        "README.md",
        "comparison-plan.json",
        "comparison-plan.schema.json",
        "gap-ledger.json",
        "gap-ledger.schema.json",
    ]


def test_leaf_analysis_records_are_owned_by_their_task_bundles() -> None:
    ledger = _load("gap-ledger.json")
    for record in ledger["analysis_records"]:
        leaf_path = ROOT / str(record["analysis_ref"])
        # The leaf is a direct child of its task bundle's analysis/ directory.
        assert leaf_path.parent.name == "analysis"
        assert leaf_path.parent.parent.name == record["task_id"]
        assert leaf_path.parent.parent.parent == DATASET
        assert leaf_path.is_file()


def test_leaf_analysis_records_carry_no_snapshot_binding() -> None:
    """Leaves are snapshot-independent.

    Embedding snapshot_id in a leaf would create a digest cycle: the leaf is
    inside the Harbor task checksum, which is itself part of the snapshot lock.
    Only the generated aggregate gap-ledger.json and comparison-plan.json bind
    the immutable snapshot.
    """
    ledger = _load("gap-ledger.json")
    ledger_schema = _load("gap-ledger.schema.json")
    leaf_validator = Draft202012Validator(
        _leaf_schema(ledger_schema), format_checker=FormatChecker()
    )
    for record in ledger["analysis_records"]:
        leaf = _load_leaf(record)
        leaf_validator.validate(leaf)
        # No snapshot binding of any shape on the leaf.
        assert "snapshot" not in leaf
        assert "snapshot_id" not in leaf
        assert "lock_path" not in leaf
        assert "snapshot_ref" not in leaf
        # The leaf's analysis content is also free of any snapshot fields.
        assert "snapshot" not in leaf["analysis"]
        assert "snapshot_id" not in leaf["analysis"]
        # The leaf's top-level keys are exactly the snapshot-independent set.
        assert set(leaf) == {"schema_version", "ledger_id", "task_id", "analysis"}
