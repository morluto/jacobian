from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "path-dependent-limit"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def test_reference_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_alternative_family_member_and_paths_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"] = {
        "exponent_p": 3,
        "function": {
            "numerator_x_power": 6,
            "numerator_y_power": 1,
            "denominator_terms": [
                {"x_power": 12, "y_power": 0},
                {"x_power": 0, "y_power": 2},
            ],
        },
        "origin_value": "0",
        "line_certificate": {
            "axes_zero": True,
            "numerator_order": 7,
            "denominator_leading_order": 2,
            "quotient_order": 5,
            "arbitrary_nonzero_slope_limit": "0",
        },
        "nonlinear_paths": [
            {"c": "3", "y_x_power": 6, "limit": "3/10"},
            {"c": "-2", "y_x_power": 6, "limit": "-2/5"},
            {"c": "1/3", "y_x_power": 6, "limit": "3/10"},
        ],
    }
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_commuted_terms_and_equivalent_rational_strings_pass(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    result = submission["result"]
    result["function"]["denominator_terms"].reverse()
    result["nonlinear_paths"] = [
        {"c": "+1", "y_x_power": 4, "limit": "0.5"},
        {"c": "2", "y_x_power": 4, "limit": "0.4"},
        {"c": "-0.5", "y_x_power": 4, "limit": "-0.4"},
    ]
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_equivalent_multivariable_nonexistence_wording_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        evidence.read_text().replace(
            "the limit at the origin does not exist",
            "no single multivariable limit exists at the origin",
        )
    )
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_unrelated_straight_line_nonexistence_wording_is_rejected(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        evidence.read_text().replace(
            "the limit at the origin does not exist",
            "no single straight-line limit exists at the origin",
        )
    )
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_nested_boolean_integer_fields_are_rejected(tmp_path: Path) -> None:
    for index in range(4):
        task, app, logs = _case(tmp_path / f"boolean-{index}")
        path = app / "submission.json"
        submission = json.loads(path.read_text())
        result = submission["result"]
        if index == 0:
            result["function"]["numerator_y_power"] = True
        elif index == 1:
            result["function"]["denominator_terms"][0]["x_power"] = False
        elif index == 2:
            result["line_certificate"]["quotient_order"] = True
        else:
            result["nonlinear_paths"][0]["y_x_power"] = False
        _fixtures._bind_result_evidence(app, submission)
        _fixtures._write_json(path, submission)
        assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_wrong_universal_line_order_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["line_certificate"]["quotient_order"] = 0
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_one_nonlinear_path_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["nonlinear_paths"] = submission["result"]["nonlinear_paths"][
        :1
    ]
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_duplicate_path_parameters_are_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["nonlinear_paths"][1] = dict(
        submission["result"]["nonlinear_paths"][0]
    )
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_wrong_path_limit_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    submission["result"]["nonlinear_paths"][0]["limit"] = "0"
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_visible_input_tampering_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    data = json.loads((app / "input.json").read_text())
    data["source"]["row"] = 674
    _fixtures._write_json(app / "input.json", data)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_keyword_only_evidence_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    path = app / "submission.json"
    submission = json.loads(path.read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text(
        "straight line nonlinear limit origin does not exist\nRESULT_JSON:"
        + json.dumps(submission["result"], separators=(",", ":"))
        + "\n"
    )
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(path, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0
