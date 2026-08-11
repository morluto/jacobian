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
    / "benchmarks/datasets/conjecture-probes-v1/zarankiewicz-projective-plane-certificate"
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


def run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_independent_projective_order_pass(tmp_path):
    app, logs, submission = case(tmp_path)
    assert run(app, logs).details["aggregate_reward"] == 1.0
    permutation = list(reversed(range(13)))
    result = submission["result"]
    result["points"] = [result["points"][i] for i in permutation]
    result["lines"] = [result["lines"][i] for i in permutation]
    inverse = {old: new for new, old in enumerate(permutation)}
    result["edges"] = sorted([[inverse[i], inverse[j]] for i, j in result["edges"]])
    write(app, submission)
    assert run(app, logs).details["aggregate_reward"] == 1.0


def test_missing_and_nonincidence_edges_fail(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["edges"][-1] = submission["result"]["edges"][0]
    write(app, submission)
    assert run(app, logs).details["aggregate_reward"] == 0.0
    app, logs, submission = case(tmp_path / "nonincidence")
    submission["result"]["edges"][0] = [0, 0]
    write(app, submission)
    assert run(app, logs).details["mathematics"] == 0.0


def test_bad_projective_normalization_and_pair_count_fail(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["result"]["points"][0] = [2, 0, 0]
    write(app, submission)
    assert run(app, logs).details["mathematics"] == 0.0
    app, logs, submission = case(tmp_path / "pair")
    submission["result"]["left_pair_common_counts"][0]["common_neighbors"] = 0
    write(app, submission)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_false_assurance_and_tampered_evidence_fail(tmp_path):
    app, logs, submission = case(tmp_path)
    submission["claimed_assurance"] = "VERIFIED"
    write(app, submission)
    result = run(app, logs)
    assert result.details["mathematics"] == 1.0 and result.details["assurance"] == 0.0
    assert result.details["aggregate_reward"] == 0.0
    app, logs, _ = case(tmp_path / "evidence")
    (app / "evidence/answer.json").write_text("tampered\n")
    result = run(app, logs)
    assert result.details["mathematics"] == 1.0 and result.details["evidence"] == 0.0
    assert result.details["aggregate_reward"] == 0.0


def test_malformed_json_preserves_fail_closed_behavior(tmp_path):
    app, logs, _ = case(tmp_path)
    (app / "submission.json").write_text('{"claimed_assurance": NaN}\n')
    result = run(app, logs)
    assert result.details["aggregate_reward"] == 0.0
    assert result.details["mathematics"] == 0.0
