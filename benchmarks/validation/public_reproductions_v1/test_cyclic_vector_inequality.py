from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.public_reproductions_v1 import support

TASK = "cyclic-vector-inequality"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert support._run_verifier(task, app, logs).reward == 1.0


def test_alternative_dimension_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    n = 7
    result = submission["result"]
    result["dimension"] = n
    submission["scope"] = f"the cyclic vector inequality at dimension n = {n}"
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
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 1.0


def test_broken_cycle_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["vectors"][2]["second_variable"] = 5
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_wrong_square_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["completed_square"]["lhs_coefficients"][2] += 1
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_nonsharp_witness_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["equality_witness"]["values"][0] = "1/3"
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_false_verified_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["claimed_assurance"] = "VERIFIED"
    support._write_json(path, submission)
    result = support._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.details["false_certification"] is True


def test_boolean_certificate_fields_are_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    for vector in submission["result"]["vectors"]:
        vector["second_constant"] = True
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_boolean_aggregate_coefficients_are_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["aggregate"]["first_constant"] = False
    submission["result"]["aggregate"]["first_coefficients"] = [
        True for _ in submission["result"]["aggregate"]["first_coefficients"]
    ]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_infinite_equality_value_is_cleanly_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["equality_witness"]["values"][0] = float("inf")
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_notably_limitation_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["limitations"] = [
        "A proof assistant verified the result; notably, all checks passed."
    ]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0


def test_hyphenated_negation_limitation_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["limitations"] = ["Not proof-assistant verified."]
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 1.0


def test_universal_scope_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["scope"] = "all real cyclic sequences of length n > 1"
    support._bind_result_evidence(app, submission)
    support._write_json(path, submission)
    assert support._run_verifier(task, app, logs).reward == 0.0
