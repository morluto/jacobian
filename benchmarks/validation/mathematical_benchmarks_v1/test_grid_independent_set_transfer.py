import importlib.util
import json
import sys
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/grid-independent-set-transfer"
)


def load_verifier():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "grid_transfer_verifier", TASK / "tests/verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _make_evidence(submission):
    return {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }


def _write_evidence_and_submission(app, submission):
    evidence_path = app / "evidence" / "answer.txt"
    evidence = _make_evidence(submission)
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)


def test_independent_transfer_derivation():
    result = load_verifier().derive()
    assert [case["independent_set_count"] for case in result["cases"]] == [
        7,
        63,
        1234,
        55447,
    ]
    assert result["total"] == 56751


def test_state_and_transition_traces_are_material():
    result = load_verifier().derive()
    assert [len(case["valid_row_masks"]) for case in result["cases"]] == [3, 5, 8, 13]
    assert [case["compatible_pair_count"] for case in result["cases"]] == [
        7,
        17,
        41,
        99,
    ]


def test_corrupt_intermediate_layer_is_rejected():
    verifier = load_verifier()
    result = verifier.derive()
    result["cases"][-1]["layer_totals"][2] += 1
    assert not verifier.matches(result)


def test_contract_has_no_verified_upgrade():
    contract = json.loads((TASK / "tests/public_contract.json").read_text())
    assert contract["allowed_assurance"] == ["COMPUTED"]


def test_accepts_computed_submission(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "grid-independent-set-transfer", "computed"
    )
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.details["evidence_validity"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_float_in_result(tmp_path: Path) -> None:
    """JSON floats in integer fields must not earn reward via Python equality."""
    task, app, logs = support._prepare_case(
        tmp_path, "grid-independent-set-transfer", "computed"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cases"][0]["independent_set_count"] = 7.0
    _write_evidence_and_submission(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_bool_in_result(tmp_path: Path) -> None:
    """JSON booleans must not pass as integers (True == 1 in Python)."""
    task, app, logs = support._prepare_case(
        tmp_path, "grid-independent-set-transfer", "computed"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cases"][0]["valid_row_masks"][1] = True
    _write_evidence_and_submission(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_float_in_evidence(tmp_path: Path) -> None:
    """Evidence with float values must not pass even if submission has integers."""
    task, app, logs = support._prepare_case(
        tmp_path, "grid-independent-set-transfer", "computed"
    )
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence = json.loads(json.dumps(_make_evidence(submission)))
    evidence["result"]["cases"][0]["independent_set_count"] = 7.0
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0
