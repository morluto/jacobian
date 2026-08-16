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
    (app / "submission.json").write_text(
        json.dumps({"result": submission["result"]}) + "\n"
    )


def run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_independent_projective_order_pass(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    assert run(app, logs).reward == 1.0
    permutation = list(reversed(range(13)))
    result = submission["result"]
    result["points"] = [result["points"][i] for i in permutation]
    result["lines"] = [result["lines"][i] for i in permutation]
    inverse = {old: new for new, old in enumerate(permutation)}
    result["edges"] = sorted([[inverse[i], inverse[j]] for i, j in result["edges"]])
    write(app, submission)
    assert run(app, logs).reward == 1.0


def test_missing_edge_and_bad_pair_count_fail(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    submission["result"]["edges"][-1] = submission["result"]["edges"][0]
    write(app, submission)
    assert run(app, logs).reward == 0.0
    app, logs, submission = case(tmp_path / "pair")
    submission["result"]["left_pair_common_counts"][0]["common_neighbors"] = 0
    write(app, submission)
    assert run(app, logs).reward == 0.0


def test_boolean_edge_count_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    submission["result"]["edge_count"] = True
    write(app, submission)
    result = run(app, logs)
    assert result.details["mathematics"] == 0.0
    assert result.reward == 0.0


def test_input_binding_fails_independently(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path / "input")
    (app / "input.json").write_text("{}\n")
    result = run(app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["mathematics"] == 1.0
    assert result.reward == 0.0


def test_malformed_submission_fails_closed(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    (app / "submission.json").write_text('{"result": NaN}\n')
    assert run(app, logs).reward == 0.0
