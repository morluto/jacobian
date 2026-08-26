from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/hadamard-order12-construction"


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    run_solution(TASK, app)
    return app, logs, json.loads((app / "submission.json").read_text())


def _write(app: Path, submission: dict) -> None:
    (app / "submission.json").write_text(
        json.dumps({"result": submission["result"]}, sort_keys=True) + "\n"
    )


def _run(app: Path, logs: Path):
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


def test_tampered_input_fails_closed(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    frozen = json.loads((app / "input.json").read_text())
    frozen["order"] = 8
    (app / "input.json").write_text(json.dumps(frozen))
    _write(app, submission)
    assert _run(app, logs).details["input_binding"] == 0.0
    assert _run(app, logs).reward == 0.0


def test_result_only_submission_is_accepted(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    assert "witness" not in submission
    _write(app, submission)
    assert _run(app, logs).reward == 1.0
