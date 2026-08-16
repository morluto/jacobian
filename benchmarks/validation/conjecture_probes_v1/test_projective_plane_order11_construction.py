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
    / "benchmarks/datasets/conjecture-probes-v1/projective-plane-order11-construction"
)


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def _write(app: Path, submission: object) -> None:
    (app / "submission.json").write_text(json.dumps(submission) + "\n")


def test_oracle_and_point_relabeling_get_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 1.0
    permutation = list(reversed(range(133)))
    submission["result"]["lines"] = [
        [permutation[point] for point in reversed(line)]
        for line in reversed(submission["result"]["lines"])
    ]
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 1.0


def test_repeated_pair_and_boolean_point_fail(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["lines"][0][0] = submission["result"]["lines"][1][0]
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
    _write(app, {"result": {"lines": [[False] * 12] * 133}})
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0


def test_input_and_extra_field_fail_closed(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    (app / "input.json").write_text("{}\n")
    result = run_verifier_in_child(task=TASK, app=app, logs=logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0
    submission["extra"] = None
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
