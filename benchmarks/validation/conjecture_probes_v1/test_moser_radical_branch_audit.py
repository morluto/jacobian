from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/moser-radical-branch-audit"


def case(tmp_path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def write(app, submission):
    payload = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence = app / "evidence/answer.json"
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")


def run(app, logs):
    output = run_verifier_in_child(task=TASK, app=app, logs=logs)
    if hasattr(output, "details"):
        return {"reward": output.reward, **output.details}
    return output


def test_oracle_passes(tmp_path):
    app, logs, _ = case(tmp_path)
    result = run(app, logs)
    assert result["correctness"] == result["mathematics"] == 1.0
    assert result["aggregate_reward"] == result["reward"] == 1.0
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert "correctness" in json.loads((logs / "reward-details.json").read_text())


def test_unordered_pair_and_edge_representations_pass(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["corrupted_pair_table"].reverse()
    submission["result"]["corrected_pair_table"].reverse()
    for table_name in ("corrupted_pair_table", "corrected_pair_table"):
        for row in submission["result"][table_name]:
            row["pair"].reverse()
    submission["result"]["false_claimed_edges"].reverse()
    submission["result"]["corrected_edges"].reverse()
    for name in ("false_claimed_edges", "corrected_edges"):
        for edge in submission["result"][name]:
            edge.reverse()
    write(app, submission)
    assert run(app, logs)["aggregate_reward"] == 1.0


def test_sign_erasure_and_incomplete_pair_audit_fail(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["false_claimed_edges"] = []
    write(app, submission)
    assert run(app, logs)["mathematics"] == 0.0
    app, logs, submission = case(tmp_path / "pair")
    submission["result"]["corrupted_pair_table"].pop()
    write(app, submission)
    assert run(app, logs)["aggregate_reward"] == 0.0


def test_noncanonical_fraction_and_false_assurance_fail(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["corrected_pair_table"][0]["distance_squared"] = ["2/2", "0"]
    write(app, submission)
    assert run(app, logs)["mathematics"] == 0.0
    app, logs, submission = case(tmp_path / "assurance")
    submission["claimed_assurance"] = "VERIFIED"
    write(app, submission)
    result = run(app, logs)
    assert (
        result["mathematics"] == 1.0
        and result["false_certification"] is True
        and result["reward"] == 0.0
    )


def test_lower_assurance_is_valid_but_receives_partial_credit(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["claimed_assurance"] = "COMPUTED"
    write(app, submission)
    result = run(app, logs)
    assert result["protocol"] == result["mathematics"] == 1.0
    assert result["assurance"] == result["aggregate_reward"] == 0.5


def test_evidence_tamper_and_malformed_json_fail(tmp_path):
    app, logs, _ = case(tmp_path)
    (app / "evidence/answer.json").write_text("tampered\n")
    result = run(app, logs)
    assert result["mathematics"] == 1.0 and result["evidence"] == 0.0
    app, logs, _ = case(tmp_path / "json")
    (app / "submission.json").write_text('{"x":NaN}\n')
    assert run(app, logs)["reward"] == 0.0
