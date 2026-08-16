from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/ramsey-r3-13-lower-bound-61"


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


def test_oracle_and_vertex_relabeling_get_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 1.0
    permutation = list(reversed(range(60)))
    submission["result"]["edges"] = [
        [permutation[left], permutation[right]]
        for left, right in reversed(submission["result"]["edges"])
    ]
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 1.0


def test_triangle_duplicate_and_boolean_endpoint_fail(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    adjacency = {tuple(sorted(edge)) for edge in submission["result"]["edges"]}
    triangle = next(
        (a, b, c)
        for a in range(60)
        for b in range(a + 1, 60)
        for c in range(b + 1, 60)
        if (a, b) in adjacency and (a, c) in adjacency and (b, c) not in adjacency
    )
    submission["result"]["edges"].append([triangle[1], triangle[2]])
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
    app, logs, submission = _case(tmp_path / "duplicate")
    submission["result"]["edges"].append(
        list(reversed(submission["result"]["edges"][0]))
    )
    _write(app, submission)
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
    _write(app, {"result": {"edges": [[False, 1]]}})
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0


def test_input_and_malformed_submission_fail_closed(tmp_path: Path) -> None:
    app, logs, _ = _case(tmp_path)
    (app / "input.json").write_text("{}\n")
    result = run_verifier_in_child(task=TASK, app=app, logs=logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0
    (app / "submission.json").write_text('{"result":NaN}\n')
    assert run_verifier_in_child(task=TASK, app=app, logs=logs).reward == 0.0
