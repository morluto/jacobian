import hashlib
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "gram-schmidt-nonzero-filter-audit"


def oracle():
    return json.loads(
        (
            Path("benchmarks/datasets/mathematical-benchmarks-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )


def verify(tmp_path, submission, *, evidence_result=None):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir()
    shutil.copy2(task / "environment/input.json", app / "input.json")
    evidence = {
        "schema_version": "1",
        "task_id": "jacobian/gram-schmidt-nonzero-filter-audit",
        "result": submission["result"] if evidence_result is None else evidence_result,
    }
    p = app / "evidence/gram-schmidt-audit.json"
    p.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(task, app, logs)


def test_oracle_and_orthogonally_transformed_system(tmp_path):
    assert verify(tmp_path / "oracle", oracle()).reward == 1.0
    alt = oracle()
    for v in alt["result"]["vectors"]:
        v[0] *= -1
    for v in alt["result"]["residuals"]:
        v[0]["numerator"] *= -1
    assert verify(tmp_path / "alt", alt).reward == 1.0


def test_reordered_zero_residual_indices_are_accepted(tmp_path):
    reordered = oracle()
    reordered["result"]["zero_residual_indices"].reverse()

    accepted = verify(tmp_path, reordered)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


@pytest.mark.parametrize(
    ("position", "replacement"),
    [(0, False), (1, True)],
    ids=("false-for-zero", "true-for-one"),
)
def test_boolean_integer_alias_in_nested_evidence_is_rejected(
    tmp_path,
    position,
    replacement,
):
    submission = oracle()
    evidence_result = json.loads(json.dumps(submission["result"]))
    evidence_result["formal_selected_indices"][position] = replacement

    rejected = verify(tmp_path, submission, evidence_result=evidence_result)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_duplicate_zero_residual_index_is_rejected(tmp_path):
    duplicate = oracle()
    duplicate["result"]["zero_residual_indices"] = [4, 4]

    rejected = verify(tmp_path, duplicate)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_malformed_boolean_and_input_tamper_fail_closed(tmp_path):
    bad = oracle()
    bad["result"]["vectors"][0][0] = True
    assert verify(tmp_path / "boolean", bad).reward == 0
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "tamper/app", tmp_path / "tamper/logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    (app / "input.json").write_text("{}")
    (app / "submission.json").write_text(json.dumps(oracle()))
    assert _run_verifier(task, app, logs).reward == 0
