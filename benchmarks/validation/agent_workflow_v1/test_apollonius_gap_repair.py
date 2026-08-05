from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "apollonius-gap-repair"


def _load(app: Path) -> dict[str, object]:
    return json.loads((app / "submission.json").read_text())


def _bind_evidence(app: Path, submission: dict[str, object]) -> None:
    result = submission["result"]
    text = (
        "\n".join(
            [
                "apollonius-coefficient-certificate-v1",
                f"multiplier: {result['multiplier']}",
                "circle_coefficients: " + ",".join(result["circle_coefficients"]),
                "distance_coefficients: " + ",".join(result["distance_coefficients"]),
            ]
        )
        + "\n"
    )
    path = app / "evidence/answer.txt"
    path.write_text(text)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_accepts_alternative_normalization(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    result = sub["result"]
    result.update(
        {
            "k": "1/2",
            "c": "4",
            "p": "4/3",
            "q": "-4",
            "center": "-4/3",
            "radius": "8/3",
            "circle_coefficients": ["1", "1", "8/3", "-16/3"],
            "distance_coefficients": ["3/4", "3/4", "2", "-4"],
            "multiplier": "3/4",
        }
    )
    _bind_evidence(app, sub)
    support._write_json(app / "submission.json", sub)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_corrupt_proportionality(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    sub["result"]["distance_coefficients"][2] = "23"
    support._write_json(app / "submission.json", sub)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_extra_result_field_is_protocol_only(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["unexpected"] = True
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 1.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_rejects_unbound_explanation(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    path = app / "evidence/answer.txt"
    path.write_text("polynomial\n")
    sub["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    support._write_json(app / "submission.json", sub)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_input_binding_is_reported_separately(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    reward = support._run_verifier(task, app, logs)
    assert reward["input_binding"] == 0.0
    assert reward["correctness"] == 1.0
    assert reward["reward"] == 0.0


def test_rejects_explosive_and_noncanonical_rationals(tmp_path: Path) -> None:
    for name, value in (("explosive", "1e999999999"), ("noncanonical", "12/1")):
        task, app, logs = support._prepare_case(tmp_path / name, TASK, "computed")
        submission = _load(app)
        submission["result"]["k"] = value
        _bind_evidence(app, submission)
        support._write_json(app / "submission.json", submission)
        reward = support._run_verifier(task, app, logs)
        assert reward["correctness"] == 0.0
        assert reward["protocol_compliance"] == 0.0


@pytest.mark.parametrize(("field", "value"), (("k", "1"), ("c", "0"), ("radius", "-1")))
def test_declared_rational_constraints_are_protocol_requirements(
    tmp_path: Path, field: str, value: str
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"][field] = value
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    reward = support._run_verifier(task, app, logs)
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_evidence_requires_four_coefficients(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["circle_coefficients"].append("0")
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_evidence_requires_string_multiplier(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["multiplier"] = ["-3"]
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_rejects_oversized_evidence(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    path = app / "evidence/answer.txt"
    path.write_text("x" * 1_000_000)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_result_shape_failure_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["unexpected"] = "ignored by the math check"
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["evidence_validity"] == 1.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_completeness_and_protocol_are_reported(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["completeness"] = "PARTIAL"
    support._write_json(app / "submission.json", submission)
    reward = support._run_verifier(task, app, logs)
    assert reward["scope_accuracy"] == 1.0
    assert reward["completeness_accuracy"] == 0.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_unsupported_assurance_is_a_protocol_failure(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(app / "submission.json", submission)

    reward = support._run_verifier(task, app, logs)
    assert reward["correctness"] == 1.0
    assert reward["assurance_calibration"] == 0.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


@pytest.mark.parametrize(
    ("field", "index"),
    (("multiplier", None), ("circle_coefficients", 0), ("distance_coefficients", 0)),
)
def test_unencodable_rational_fails_closed(
    tmp_path: Path, field: str, index: int | None
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    if index is None:
        submission["result"][field] = "\ud800"
    else:
        submission["result"][field][index] = "\ud800"
    support._write_json(app / "submission.json", submission)

    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 0.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0


def test_evidence_lines_reject_extra_spaces(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    path = app / "evidence/answer.txt"
    path.write_text(" " + path.read_text())
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    support._write_json(app / "submission.json", submission)

    reward = support._run_verifier(task, app, logs)
    assert reward["evidence_validity"] == 0.0
    assert reward["reward"] == 0.0


def test_extra_limitation_is_reported_in_its_own_dimension(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["limitations"].append("extra")
    support._write_json(app / "submission.json", submission)

    reward = support._run_verifier(task, app, logs)
    assert reward["limitations_accuracy"] == 0.0
    assert reward["protocol_compliance"] == 0.0
    assert reward["reward"] == 0.0
