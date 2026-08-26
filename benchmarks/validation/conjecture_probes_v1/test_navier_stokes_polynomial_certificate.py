from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/conjecture-probes-v1/navier-stokes-polynomial-certificate"
)


def _q(value) -> dict[str, int]:
    parsed = Fraction(value)
    return {"numerator": parsed.numerator, "denominator": parsed.denominator}


def _qs(values: list[object]) -> list[dict[str, int]]:
    return [_q(item) for item in values]


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app, logs = (tmp_path / "app", tmp_path / "logs")
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    run_solution(TASK, app)
    return (app, logs, json.loads((app / "submission.json").read_text()))


def _write(app: Path, submission: dict) -> None:
    submission = dict(submission)
    submission.pop("witness", None)
    (app / "submission.json").write_text(
        __import__("json").dumps({"result": submission["result"]}) + "\n"
    )


def _run(app: Path, logs: Path) -> dict:
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_alternative_scaled_rotation_pass(tmp_path: Path) -> None:
    app, logs, _submission = _case(tmp_path)
    assert _run(app, logs).reward == 1.0
    app, logs, submission = _case(tmp_path / "alt")
    submission["result"] = {
        "velocity": [_qs(["0", "0", "-2"]), _qs(["0", "2", "0"])],
        "pressure": _qs(["0", "0", "0", "2", "0", "2"]),
        "divergence": _qs(["0"]),
        "momentum_x": _qs(["0", "0", "0"]),
        "momentum_y": _qs(["0", "0", "0"]),
        "vorticity": _q("4"),
    }
    _write(app, submission)
    assert _run(app, logs).reward == 1.0


def test_zero_field_and_wrong_residual_fail(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["velocity"] = [_qs(["0"] * 3), _qs(["0"] * 3)]
    submission["result"]["pressure"] = _qs(["0"] * 6)
    submission["result"]["vorticity"] = _q("0")
    _write(app, submission)
    assert _run(app, logs).details["correctness"] == 0.0
    app, logs, submission = _case(tmp_path / "residual")
    submission["result"]["momentum_x"][1] = _q("1")
    _write(app, submission)
    assert _run(app, logs).reward == 0.0


def test_tampered_input_is_diagnostic_only_for_math(tmp_path: Path) -> None:
    app, logs, _submission = _case(tmp_path)
    (app / "input.json").write_text("{}\n")
    reward = _run(app, logs)
    assert reward.details["input_binding"] == 0.0
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 0.0
