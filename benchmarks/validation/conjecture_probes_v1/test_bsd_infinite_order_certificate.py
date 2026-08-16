from __future__ import annotations

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


def write(app: Path, s: dict) -> None:
    (app / "submission.json").write_text(json.dumps({"result": s["result"]}) + "\n")


def run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path):
    app, logs, _ = case(tmp_path)
    assert run(app, logs).details["aggregate_reward"] == 1.0


def test_wrong_multiple_and_divisibility_label_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["double"][0] = "0"
    write(app, s)
    assert run(app, logs).details["mathematics"] == 0.0
    app, logs, s = case(tmp_path / "label")
    s["result"]["y_square_divides_discriminant"] = True
    write(app, s)
    assert run(app, logs).details["aggregate_reward"] == 0.0


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
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_tampered_input_fails_closed(tmp_path):
    app, logs, _ = case(tmp_path / "input")
    (app / "input.json").write_text("{}\n")
    reward = run(app, logs)
    assert reward.details["input_binding"] == 0.0
    assert reward.details["mathematics"] == 1.0
    assert reward.details["aggregate_reward"] == 0.0


def test_unused_evidence_file_does_not_affect_reward(tmp_path):
    app, logs, s = case(tmp_path)
    write(app, s)
    (app / "evidence").mkdir(exist_ok=True)
    (app / "evidence/answer.txt").write_text("not-json")
    verdict = run(app, logs)
    assert verdict.details["aggregate_reward"] == 1.0


def test_undeclared_witness_key_is_rejected(tmp_path):
    app, logs, s = case(tmp_path)
    s["witness"] = [{"path": "evidence/answer.txt", "sha256": "sha256:" + "0" * 64}]
    (app / "submission.json").write_text(json.dumps(s) + "\n")
    assert run(app, logs).details["aggregate_reward"] == 0.0
