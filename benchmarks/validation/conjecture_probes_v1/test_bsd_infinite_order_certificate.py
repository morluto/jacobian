from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/bsd-infinite-order-certificate"


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


def write(app: Path, s: dict):
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


def run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path):
    app, logs, _ = case(tmp_path)
    assert run(app, logs)["aggregate_reward"] == 1.0


def test_wrong_multiple_and_divisibility_label_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["double"][0] = "0"
    write(app, s)
    assert run(app, logs)["mathematics"] == 0.0
    app, logs, s = case(tmp_path / "label")
    s["result"]["y_square_divides_discriminant"] = True
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 0.0


def test_singular_curve_and_torsion_shortcut_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"].update(
        {
            "A": 0,
            "B": 0,
            "point": [0, 0],
            "discriminant": 0,
            "y_square": 0,
            "double": ["0", "0"],
            "triple": ["0", "0"],
        }
    )
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 0.0


def test_false_verified_and_tampered_input_fail_closed(tmp_path):
    app, logs, s = case(tmp_path)
    s["claimed_assurance"] = "VERIFIED"
    write(app, s)
    assert run(app, logs)["false_certification"] is True
    app, logs, _ = case(tmp_path / "input")
    (app / "input.json").write_text("{}\n")
    reward = run(app, logs)
    assert (
        reward["input_binding"] == 0.0
        and reward["mathematics"] == 1.0
        and reward["aggregate_reward"] == 0.0
    )
