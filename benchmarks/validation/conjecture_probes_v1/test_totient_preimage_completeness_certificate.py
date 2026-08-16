from __future__ import annotations

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


def write(app: Path, submission: dict) -> None:
    (app / "submission.json").write_text(
        json.dumps({"result": submission["result"]}) + "\n"
    )


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


def test_boolean_totient_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    submission["result"]["solutions"][0]["totient"] = True
    write(app, submission)
    result = run(app, logs)
    assert result.details["mathematics"] == 0.0
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


def test_unused_evidence_file_does_not_affect_reward(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    write(app, submission)
    (app / "evidence").mkdir(exist_ok=True)
    (app / "evidence/answer.json").write_text("tampered\n")
    assert run(app, logs).reward == 1.0
