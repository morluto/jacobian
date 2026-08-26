from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT / "benchmarks/datasets/conjecture-probes-v1/constant-weight-code-a23-6-10-2992"
)


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    run_solution(TASK, app)
    return app, logs, json.loads((app / "submission.json").read_text())


def _write(app: Path, submission: object) -> None:
    (app / "submission.json").write_text(json.dumps(submission) + "\n")


def test_oracle_and_coordinate_permutation_get_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 1.0
    words = submission["result"]["codewords"]
    submission["result"]["codewords"] = [
        f"{int(f'{int(word, 16):023b}'[::-1], 2):06x}" for word in reversed(words)
    ]
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 1.0


def test_wrong_distance_and_wrong_types_fail(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["codewords"][1] = submission["result"]["codewords"][0]
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
    _write(app, {"result": {"codewords": [True] * 2992}})
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0


def test_input_and_protocol_tampering_fail_closed(tmp_path: Path) -> None:
    app, logs, _ = _case(tmp_path)
    (app / "input.json").write_text("{}\n")
    result = run_verifier_in_child(task=TASK, app=app, logs=logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0
    (app / "submission.json").write_text('{"result":NaN}\n')
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
