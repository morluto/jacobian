from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/conjecture-probes-v1/yang-mills-gauge-invariance-certificate"
)


def case(tmp_path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    run_solution(TASK, app)
    return app, logs, json.loads((app / "submission.json").read_text())


def write(app, s):
    (app / "submission.json").write_text(json.dumps({"result": s["result"]}) + "\n")


def run(app, logs):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path):
    app, logs, _ = case(tmp_path)
    assert run(app, logs).details["aggregate_reward"] == 1.0


def test_accepts_unreduced_link_components(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["links"][0] = [
        {"numerator": 6, "denominator": 10},
        {"numerator": 8, "denominator": 10},
        {"numerator": 0, "denominator": 1},
        {"numerator": 0, "denominator": 1},
    ]
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 1.0


def test_corrupt_transformed_link_and_plaquette_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["transformed_links"][0][0] = {"numerator": 2, "denominator": 1}
    write(app, s)
    assert run(app, logs).details["mathematics"] == 0.0
    app, logs, s = case(tmp_path / "p")
    s["result"]["plaquette"][1] = {"numerator": 0, "denominator": 1}
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_identity_shortcut_and_nonunit_gauge_fail(tmp_path):
    app, logs, s = case(tmp_path)
    one = {"numerator": 1, "denominator": 1}
    zero = {"numerator": 0, "denominator": 1}
    s["result"]["links"] = [[one, zero, zero, zero]] * 4
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0
    app, logs, s = case(tmp_path / "unit")
    s["result"]["gauges"][0] = [
        {"numerator": 1, "denominator": 1},
        {"numerator": 1, "denominator": 1},
        {"numerator": 0, "denominator": 1},
        {"numerator": 0, "denominator": 1},
    ]
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_tampered_input_fails(tmp_path):
    app, logs, _ = case(tmp_path / "input")
    (app / "input.json").write_text("{}\n")
    r = run(app, logs)
    assert (
        r.details["input_binding"] == 0.0
        and r.details["mathematics"] == 1.0
        and r.details["aggregate_reward"] == 0.0
    )


def test_result_only_submission_is_accepted(tmp_path):
    app, logs, s = case(tmp_path)
    assert "witness" not in s
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 1.0
