from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/conjecture-probes-v1/totient-preimage-completeness-certificate"
)


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
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")


def run(app, logs):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_reordered_solutions_pass(tmp_path):
    app, logs, submission = case(tmp_path)
    assert run(app, logs)["aggregate_reward"] == 1.0
    submission["result"]["solutions"].reverse()
    write(app, submission)
    assert run(app, logs)["aggregate_reward"] == 1.0


def test_omission_extra_and_bad_factorization_fail(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["solutions"].pop()
    write(app, submission)
    assert run(app, logs)["mathematics"] == 0.0
    app, logs, submission = case(tmp_path / "factor")
    submission["result"]["solutions"][0]["factorization"] = [[2, 1], [13, 1]]
    write(app, submission)
    assert run(app, logs)["aggregate_reward"] == 0.0


def test_incomplete_prime_options_and_false_assurance_fail(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["prime_power_options"][0]["exponents"].pop()
    write(app, submission)
    assert run(app, logs)["mathematics"] == 0.0
    app, logs, submission = case(tmp_path / "assurance")
    submission["claimed_assurance"] = "VERIFIED"
    write(app, submission)
    result = run(app, logs)
    assert (
        result["mathematics"] == 1.0
        and result["assurance"] == 0.0
        and result["aggregate_reward"] == 0.0
    )


def test_evidence_tamper_and_malformed_json_fail(tmp_path):
    app, logs, _ = case(tmp_path)
    (app / "evidence/answer.txt").write_text("tampered\n")
    result = run(app, logs)
    assert result["mathematics"] == 1.0 and result["evidence"] == 0.0
    app, logs, _ = case(tmp_path / "json")
    (app / "submission.json").write_text('{"claimed_assurance":NaN}\n')
    assert run(app, logs)["aggregate_reward"] == 0.0
