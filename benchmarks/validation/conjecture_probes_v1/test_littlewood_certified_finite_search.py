from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT / "benchmarks/datasets/conjecture-probes-v1/littlewood-certified-finite-search"
)


def case(tmp_path):
    app, logs = (tmp_path / "app", tmp_path / "logs")
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    run_solution(TASK, app)
    return (app, logs, json.loads((app / "submission.json").read_text()))


def write(app, s):
    s = dict(s)
    s.pop("witness", None)
    (app / "submission.json").write_text(
        __import__("json").dumps({"result": s["result"]}) + "\n"
    )


def run(app, logs):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path):
    app, logs, _ = case(tmp_path)
    assert run(app, logs).details["aggregate_reward"] == 1.0


def test_missing_record_and_wrong_argmin_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["records"].pop()
    write(app, s)
    assert run(app, logs).details["mathematics"] == 0.0
    app, logs, s = case(tmp_path / "arg")
    s["result"]["argmin_n"] = 1
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_corrupt_bound_and_floor_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["records"][0]["upper"] = "1"
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0
    app, logs, s = case(tmp_path / "floor")
    s["result"]["records"][0]["floors"][0] += 1
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0
