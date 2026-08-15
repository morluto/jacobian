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
    / "benchmarks/datasets/conjecture-probes-v1/navier-stokes-polynomial-certificate"
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


def _write(app: Path, submission: dict) -> None:
    payload = {
        "schema_version": "1",
        "result": submission["result"],
    }
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")


def _run(app: Path, logs: Path) -> dict:
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_alternative_scaled_rotation_pass(tmp_path: Path) -> None:
    app, logs, _submission = _case(tmp_path)
    assert _run(app, logs).reward == 1.0
    app, logs, submission = _case(tmp_path / "alt")
    submission["result"] = {
        "velocity": [["0", "0", "-2"], ["0", "2", "0"]],
        "pressure": ["0", "0", "0", "2", "0", "2"],
        "divergence": ["0"],
        "momentum_x": ["0", "0", "0"],
        "momentum_y": ["0", "0", "0"],
        "vorticity": "4",
    }
    _write(app, submission)
    assert _run(app, logs).reward == 1.0


def test_zero_field_and_wrong_residual_fail(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["velocity"] = [["0"] * 3, ["0"] * 3]
    submission["result"]["pressure"] = ["0"] * 6
    submission["result"]["vorticity"] = "0"
    _write(app, submission)
    assert _run(app, logs).details["correctness"] == 0.0
    app, logs, submission = _case(tmp_path / "residual")
    submission["result"]["momentum_x"][1] = "1"
    _write(app, submission)
    assert _run(app, logs).reward == 0.0


def test_tampered_input_is_diagnostic_only_for_math(tmp_path: Path) -> None:
    app, logs, _submission = _case(tmp_path)
    (app / "input.json").write_text("{}\n")
    reward = _run(app, logs)
    assert reward.details["input_binding"] == 0.0
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 0.0
