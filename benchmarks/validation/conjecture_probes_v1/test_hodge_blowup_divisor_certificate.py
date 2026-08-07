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
    ROOT / "benchmarks/datasets/conjecture-probes-v1/hodge-blowup-divisor-certificate"
)


def case(tmp_path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def write(app, s):
    payload = {
        "schema_version": "1",
        "task_id": s["task_id"],
        "result": s["result"],
        "limitations": s["limitations"],
    }
    e = app / "evidence/answer.txt"
    e.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")


def run(app, logs):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_negative_cubic_pass(tmp_path):
    app, logs, _ = case(tmp_path)
    assert run(app, logs)["aggregate_reward"] == 1.0
    app, logs, s = case(tmp_path / "neg")
    s["result"]["coefficients"] = [-x for x in s["result"]["coefficients"]]
    for row in s["result"]["point_checks"]:
        row["gradient"] = [-x for x in row["gradient"]]
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 1.0


def test_nonvanishing_point_and_fake_gradient_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["coefficients"][9] = 1
    write(app, s)
    assert run(app, logs)["mathematics"] == 0.0
    app, logs, s = case(tmp_path / "g")
    s["result"]["point_checks"][0]["gradient"] = [0, 0, 0]
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 0.0


def test_scaled_polynomial_and_wrong_intersection_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["coefficients"] = [2 * x for x in s["result"]["coefficients"]]
    for row in s["result"]["point_checks"]:
        row["gradient"] = [2 * x for x in row["gradient"]]
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 0.0
    app, logs, s = case(tmp_path / "i")
    s["result"]["self_intersection"] = 4
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 0.0


def test_false_verified_and_tampered_input_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["claimed_assurance"] = "VERIFIED"
    write(app, s)
    assert run(app, logs)["false_certification"] is True
    app, logs, _ = case(tmp_path / "input")
    (app / "input.json").write_text("{}\n")
    r = run(app, logs)
    assert (
        r["input_binding"] == 0.0
        and r["mathematics"] == 1.0
        and r["aggregate_reward"] == 0.0
    )
