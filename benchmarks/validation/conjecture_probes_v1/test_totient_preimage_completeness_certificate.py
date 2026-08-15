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
TASK_ID = "jacobian/totient-preimage-completeness-certificate"


def case(tmp_path: Path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def _payload(result: object) -> dict[str, object]:
    return {"schema_version": "1", "task_id": TASK_ID, "result": result}


def write(app: Path, submission: dict, *, payload: object | None = None) -> None:
    evidence = app / "evidence/answer.json"
    evidence.write_text(
        json.dumps(
            _payload(submission["result"]) if payload is None else payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")


def run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_equivalent_representations_pass(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    assert run(app, logs).reward == 1.0
    submission["result"]["solutions"].reverse()
    submission["result"]["candidate_primes"].reverse()
    submission["result"]["prime_power_options"].reverse()
    submission["result"]["solutions"][0]["factorization"].reverse()
    write(app, submission)
    assert run(app, logs).reward == 1.0

    app, logs, submission = case(tmp_path / "minimal")
    for optional in (
        "candidate_primes",
        "prime_power_options",
        "enumerated_branch_count",
    ):
        submission["result"].pop(optional)
    write(app, submission)
    assert run(app, logs).reward == 1.0


def test_incomplete_classification_and_wrong_types_fail(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    submission["result"]["solutions"].pop()
    write(app, submission)
    assert run(app, logs).details["mathematics"] == 0.0
    assert run(app, logs).reward == 0.0

    app, logs, submission = case(tmp_path / "type")
    submission["result"]["solutions"][0]["factorization"][0][1] = 1.0
    write(app, submission)
    assert run(app, logs).reward == 0.0


def test_witness_is_digest_bound_and_type_strict(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    (app / "evidence/answer.json").write_text("tampered\n")
    result = run(app, logs)
    assert result.details["mathematics"] == 1.0
    assert result.details["witness_validity"] == 0.0
    assert result.reward == 0.0

    app, logs, submission = case(tmp_path / "typed")
    payload = _payload(submission["result"])
    payload["result"] = json.loads(json.dumps(payload["result"]))
    payload["result"]["solutions"][0]["totient"] = True
    write(app, submission, payload=payload)
    result = run(app, logs)
    assert result.details["mathematics"] == 1.0
    assert result.details["witness_validity"] == 0.0
    assert result.reward == 0.0


def test_malformed_submission_and_tampered_input_fail(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    (app / "submission.json").write_text('{"result":{},"result":{}}\n')
    assert run(app, logs).reward == 0.0

    app, logs, _ = case(tmp_path / "input")
    (app / "input.json").write_text("{}\n")
    result = run(app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["mathematics"] == 1.0
    assert result.reward == 0.0


def test_witness_accepts_large_legal_whitespace_padding(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    evidence = app / "evidence/answer.json"
    evidence.write_text(
        " " * (17 * 1024 * 1024)
        + json.dumps(
            _payload(submission["result"]), sort_keys=True, separators=(",", ":")
        )
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")
    assert run(app, logs).reward == 1.0
