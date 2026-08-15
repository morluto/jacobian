from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/hadamard-order12-construction"


TASK_ID = "jacobian/hadamard-order12-construction"
LIMITATIONS = ["ORDER_12_ONLY", "NO_GENERAL_HADAMARD_CONJECTURE_CONCLUSION"]


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def _write(app: Path, submission: dict) -> None:
    evidence = app / "evidence/answer.txt"
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": submission["result"],
    }
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


def _run(app: Path, logs: Path) -> dict:
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_certificate_gets_full_reward(tmp_path: Path) -> None:
    app, logs, _ = _case(tmp_path)
    assert _run(app, logs).reward == 1.0


def test_equivalent_normalized_row_column_permutation_passes(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    matrix = submission["result"]["matrix"]
    matrix[1], matrix[2] = matrix[2], matrix[1]
    for row in matrix:
        row[1], row[2] = row[2], row[1]
    _write(app, submission)
    assert _run(app, logs).reward == 1.0


def test_nonorthogonal_entry_and_wrong_determinant_fail(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["matrix"][4][7] *= -1
    _write(app, submission)
    assert _run(app, logs).details["correctness"] == 0.0

    app, logs, submission = _case(tmp_path / "det")
    submission["result"]["determinant"] += 1
    _write(app, submission)
    assert _run(app, logs).reward == 0.0


def test_unnormalized_first_row_fails(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["matrix"][0] = [
        -value for value in submission["result"]["matrix"][0]
    ]
    _write(app, submission)
    assert _run(app, logs).details["correctness"] == 0.0


def test_input_and_evidence_tampering_fail_closed(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    frozen = json.loads((app / "input.json").read_text())
    frozen["order"] = 8
    (app / "input.json").write_text(json.dumps(frozen))
    _write(app, submission)
    assert _run(app, logs).details["input_binding"] == 0.0
    assert _run(app, logs).reward == 0.0

    app, logs, submission = _case(tmp_path / "evidence")
    (app / "evidence/answer.txt").write_text("{}\n")
    reward = _run(app, logs)
    assert reward.details["witness_validity"] == 0.0
    assert reward.reward == 0.0


def test_malformed_witness_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    evidence = app / "evidence/answer.txt"
    evidence.write_text(json.dumps({"task_id": TASK_ID}) + "\n")
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")
    reward = _run(app, logs)
    assert reward.details["witness_validity"] == 0.0
    assert reward.reward == 0.0


def test_witness_result_binding_is_type_strict(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    evidence = app / "evidence/answer.txt"
    payload = json.loads(evidence.read_text())
    payload["result"]["matrix"][0][0] = True
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")

    reward = _run(app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.details["witness_validity"] == 0.0
    assert reward.reward == 0.0
