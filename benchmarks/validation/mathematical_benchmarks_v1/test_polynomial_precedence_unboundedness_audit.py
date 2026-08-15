import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "polynomial-precedence-unboundedness-audit"


def r(n, d=1):
    return {"numerator": n, "denominator": d}


def oracle():
    result = {
        "x_coefficients": [r(-7), r(0), r(-1, 2)],
        "y_coefficients": [r(0), r(1)],
        "formal_coefficients": [r(49), r(28), r(-3), r(0), r(-1, 4)],
        "checkpoints": [
            {"t": 0, "value": r(49)},
            {"t": 1, "value": r(295, 4)},
            {"t": 2, "value": r(89)},
            {"t": 5, "value": r(-169, 4)},
        ],
        "formal_status": "UNBOUNDED_BELOW",
    }
    return {
        "result": result,
        "witness": [{"path": "evidence/precedence-audit.json", "sha256": ""}],
    }


def verify(tmp_path, submission):
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "app", tmp_path / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir()
    shutil.copy2(task / "environment/input.json", app / "input.json")
    evidence = {
        "schema_version": "1",
        "task_id": f"jacobian/{TASK}",
        "result": submission["result"],
    }
    path = app / "evidence/precedence-audit.json"
    path.write_text(json.dumps(evidence, separators=(",", ":")))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission))
    return _run_verifier(task, app, logs)


def test_oracle_and_alternative_family(tmp_path):
    assert verify(tmp_path / "oracle", oracle()).reward == 1.0
    alt = oracle()
    # x=-t^2/4-7, y=t gives leading coefficient -3/16.
    alt["result"] = {
        "x_coefficients": [r(-7), r(0), r(-1, 4)],
        "y_coefficients": [r(0), r(1)],
        "formal_coefficients": [r(49), r(28), r(-3), r(0), r(-3, 16)],
        "checkpoints": [
            {"t": 0, "value": r(49)},
            {"t": 1, "value": r(1181, 16)},
            {"t": 2, "value": r(90)},
            {"t": 4, "value": r(65)},
        ],
        "formal_status": "UNBOUNDED_BELOW",
    }
    assert verify(tmp_path / "alt", alt).reward == 1.0


def test_corruption_fails(tmp_path):
    bad = oracle()
    bad["result"]["formal_coefficients"][-1] = r(-1, 3)
    assert verify(tmp_path / "bad", bad).reward == 0


def test_malformed_and_input_tamper_fail_closed(tmp_path):
    malformed = oracle()
    malformed["result"]["x_coefficients"][0]["numerator"] = True
    assert verify(tmp_path / "malformed", malformed).reward == 0
    task = Path("benchmarks/datasets/mathematical-benchmarks-v1") / TASK
    app, logs = tmp_path / "tamper/app", tmp_path / "tamper/logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    (app / "input.json").write_text("{}")
    (app / "submission.json").write_text(json.dumps(oracle()))
    assert _run_verifier(task, app, logs).reward == 0
