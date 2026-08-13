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


def test_oracle_and_equivalent_representations_pass(tmp_path):
    app, logs, submission = case(tmp_path)
    oracle = run(app, logs)
    assert oracle["correctness"] == oracle["mathematics"] == 1.0
    assert oracle["aggregate_reward"] == 1.0
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
    assert "correctness" in json.loads((logs / "reward-details.json").read_text())
    submission["result"]["solutions"].reverse()
    submission["result"]["candidate_primes"].reverse()
    submission["result"]["prime_power_options"].reverse()
    submission["result"]["solutions"][0]["factorization"].reverse()
    write(app, submission)
    assert run(app, logs)["aggregate_reward"] == 1.0

    app, logs, submission = case(tmp_path / "minimal")
    for optional in (
        "candidate_primes",
        "prime_power_options",
        "enumerated_branch_count",
    ):
        submission["result"].pop(optional)
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


def test_unknown_result_field_only_fails_protocol(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["notes"] = "schema-disallowed annotation"
    write(app, submission)
    result = run(app, logs)
    assert result["mathematics"] == result["correctness"] == 1.0
    assert result["protocol"] == result["aggregate_reward"] == 0.0


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

    app, logs, submission = case(tmp_path / "unhashable-assurance")
    submission["claimed_assurance"] = []
    write(app, submission)
    result = run(app, logs)
    assert (
        result["correctness"] == result["mathematics"] == 1.0
        and result["protocol"] == result["assurance"] == 0.0
        and result["aggregate_reward"] == 0.0
    )


def test_integral_floats_in_optional_and_required_certificates_fail(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["candidate_primes"][0] = 2.0
    write(app, submission)
    result = run(app, logs)
    assert result["correctness"] == result["mathematics"] == 0.0
    assert result["aggregate_reward"] == 0.0

    app, logs, submission = case(tmp_path / "factor")
    submission["result"]["solutions"][0]["factorization"][0][1] = 1.0
    write(app, submission)
    assert run(app, logs)["mathematics"] == 0.0


def test_evidence_tamper_and_malformed_json_fail(tmp_path):
    app, logs, _ = case(tmp_path)
    (app / "evidence/answer.json").write_text("tampered\n")
    result = run(app, logs)
    assert result["mathematics"] == 1.0 and result["evidence"] == 0.0


def test_evidence_copy_uses_json_typed_equality(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["limitations"] = [1]
    evidence = app / "evidence/answer.json"
    payload = json.loads(evidence.read_text())
    payload["limitations"] = [True]
    evidence.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")

    result = run(app, logs)
    assert result["mathematics"] == 1.0
    assert result["evidence"] == result["aggregate_reward"] == 0.0


def test_raw_submission_rejects_duplicate_keys(tmp_path):
    app, logs, submission = case(tmp_path)
    canonical = json.dumps(submission["result"], separators=(",", ":"))
    (app / "submission.json").write_text(
        '{"result":{"accepted_count":0},"result":' + canonical + "}"
    )

    result = run(app, logs)
    assert result["mathematics"] == result["aggregate_reward"] == 0.0


def test_evidence_requires_present_string_task_id(tmp_path):
    app, logs, submission = case(tmp_path)
    submission.pop("task_id")
    evidence = app / "evidence/answer.json"
    payload = json.loads(evidence.read_text())
    payload["task_id"] = None
    evidence.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")

    result = run(app, logs)
    assert result["mathematics"] == 1.0
    assert result["evidence"] == result["aggregate_reward"] == 0.0
    app, logs, _ = case(tmp_path / "json")
    (app / "submission.json").write_text('{"claimed_assurance":NaN}\n')
    assert run(app, logs)["aggregate_reward"] == 0.0

    app, logs, submission = case(tmp_path / "nan-evidence")
    evidence = app / "evidence/answer.json"
    evidence.write_text('{"schema_version":"1","task_id":NaN}')
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")
    result = run(app, logs)
    assert result["mathematics"] == 1.0 and result["evidence"] == 0.0


def test_unknown_solution_does_not_collapse_independent_diagnostics(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["solutions"][0] = {
        "n": 1,
        "factorization": [],
        "totient": 48,
    }
    submission["claimed_assurance"] = "VERIFIED"
    write(app, submission)

    result = run(app, logs)

    assert result["correctness"] == 0.0
    assert result["false_certification"] == 1.0


def test_evidence_accepts_large_legal_whitespace_padding(tmp_path):
    app, logs, submission = case(tmp_path)
    evidence_path = app / "evidence/answer.json"
    payload = evidence_path.read_text()
    evidence_path.write_text(" " * (17 * 1024 * 1024) + payload)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")

    assert run(app, logs)["aggregate_reward"] == 1.0
