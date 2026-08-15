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
    ROOT
    / "benchmarks/datasets/conjecture-probes-v1/yang-mills-gauge-invariance-certificate"
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
        "task_id": "jacobian/yang-mills-gauge-invariance-certificate",
        "result": s["result"],
    }
    e = app / "evidence/answer.txt"
    e.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    s["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")


def run(app, logs):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path):
    app, logs, _ = case(tmp_path)
    assert run(app, logs).details["aggregate_reward"] == 1.0


def test_corrupt_transformed_link_and_plaquette_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["transformed_links"][0][0] = "2"
    write(app, s)
    assert run(app, logs).details["mathematics"] == 0.0
    app, logs, s = case(tmp_path / "p")
    s["result"]["plaquette"][1] = "0"
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_identity_shortcut_and_nonunit_gauge_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["links"] = [["1", "0", "0", "0"]] * 4
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0
    app, logs, s = case(tmp_path / "unit")
    s["result"]["gauges"][0] = ["1", "1", "0", "0"]
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
