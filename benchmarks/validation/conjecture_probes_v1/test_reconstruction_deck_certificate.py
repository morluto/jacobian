from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/reconstruction-deck-certificate"


def case(tmp_path: Path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    run_solution(TASK, app)
    return app, logs, json.loads((app / "submission.json").read_text())


def write(app: Path, submission: dict) -> None:
    (app / "submission.json").write_text(
        json.dumps({"result": submission["result"]}) + "\n"
    )


def run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    result = run(app, logs)
    assert result.reward == result.details["aggregate_reward"] == 1.0


def test_wrong_reconstruction_and_duplicate_card_fail(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    submission["result"]["embeddings"][0]["local_to_original"][:2] = reversed(
        submission["result"]["embeddings"][0]["local_to_original"][:2]
    )
    write(app, submission)
    assert run(app, logs).details["aggregate_reward"] == 0.0

    app, logs, submission = case(tmp_path / "duplicate")
    submission["result"]["embeddings"][-1] = submission["result"]["embeddings"][0]
    write(app, submission)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_tampered_input_fails_independently(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    (app / "input.json").write_text("{}\n")
    result = run(app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["mathematics"] == 1.0
    assert result.reward == 0.0


def test_result_only_submission_is_accepted(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    assert "witness" not in submission
    write(app, submission)
    assert run(app, logs).reward == 1.0


def test_extra_witness_key_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    submission["witness"] = []
    (app / "submission.json").write_text(json.dumps(submission) + "\n")
    assert run(app, logs).reward == 0.0


def test_malformed_submission_is_rejected(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    (app / "submission.json").write_text('{"result":{},"result":{}}\n')
    assert run(app, logs).reward == 0.0
