from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/hadamard-order664-construction"


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    run_solution(TASK, app)
    return app, logs, json.loads((app / "submission.json").read_text())


def _write(app: Path, submission: object) -> None:
    (app / "submission.json").write_text(json.dumps(submission) + "\n")


def test_oracle_and_equivalent_sign_permutation_get_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 1.0
    rows = list(reversed(submission["result"]["rows"]))
    rows[0] = "".join("1" if bit == "0" else "0" for bit in rows[0])
    rows = [row[1:] + row[:1] for row in rows]
    submission["result"]["rows"] = rows
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 1.0


def test_nonorthogonal_and_wrong_shape_fail(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    row = submission["result"]["rows"][1]
    submission["result"]["rows"][1] = ("1" if row[0] == "0" else "0") + row[1:]
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
    _write(app, {"result": {"rows": [True] * 664}})
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0


def test_input_and_oversized_submission_fail_closed(tmp_path: Path) -> None:
    app, logs, _ = _case(tmp_path)
    (app / "input.json").write_text("{}\n")
    result = run_verifier_in_child(task=TASK, app=app, logs=logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0
    (app / "submission.json").write_text(" " * (16 * 1024 * 1024 + 1))
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
