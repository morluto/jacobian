from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.public_reproductions_v1._fixtures import (
    _bind_result_evidence,
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

TASK = "cyclic-vector-inequality"


def _case(tmp_path: Path):
    return _prepare_case(tmp_path, TASK, "computed")


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert _run_verifier(task, app, logs).reward == 1.0


def test_alternative_dimension_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    n = 7
    result = submission["result"]
    result["dimension"] = n
    result["vectors"] = [
        {
            "index": i,
            "first_variable": i,
            "second_constant": 1,
            "second_variable": i % n + 1,
        }
        for i in range(1, n + 1)
    ]
    result["aggregate"] = {
        "first_constant": 0,
        "first_coefficients": [1] * n,
        "second_constant": n,
        "second_coefficients": [-1] * n,
    }
    result["completed_square"] = {
        "lhs_coefficients": [4, -4 * n, n * n],
        "square_coefficients": [2, -n],
    }
    result["equality_witness"]["values"] = ["1/2"] * n
    _bind_result_evidence(app, submission)
    _write_json(path, submission)
    assert _run_verifier(task, app, logs).reward == 1.0


def test_broken_cycle_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["vectors"][2]["second_variable"] = 5
    _bind_result_evidence(app, submission)
    _write_json(path, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_wrong_square_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["completed_square"]["lhs_coefficients"][2] += 1
    _bind_result_evidence(app, submission)
    _write_json(path, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_nonsharp_witness_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["equality_witness"]["values"][0] = "1/3"
    _bind_result_evidence(app, submission)
    _write_json(path, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_boolean_certificate_fields_are_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    for vector in submission["result"]["vectors"]:
        vector["second_constant"] = True
    _bind_result_evidence(app, submission)
    _write_json(path, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_boolean_aggregate_coefficients_are_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["aggregate"]["first_constant"] = False
    submission["result"]["aggregate"]["first_coefficients"] = [
        True for _ in submission["result"]["aggregate"]["first_coefficients"]
    ]
    _bind_result_evidence(app, submission)
    _write_json(path, submission)
    assert _run_verifier(task, app, logs).reward == 0.0


def test_infinite_equality_value_is_cleanly_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["equality_witness"]["values"][0] = float("inf")
    _bind_result_evidence(app, submission)
    _write_json(path, submission)
    assert _run_verifier(task, app, logs).reward == 0.0
