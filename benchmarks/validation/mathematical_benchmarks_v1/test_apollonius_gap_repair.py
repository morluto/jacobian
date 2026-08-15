from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "apollonius-gap-repair"


def _q(value) -> dict[str, int]:
    parsed = Fraction(value)
    return {"numerator": parsed.numerator, "denominator": parsed.denominator}


def _qs(values: list[object]) -> list[dict[str, int]]:
    return [_q(item) for item in values]


def _load(app: Path) -> dict[str, object]:
    return json.loads((app / "submission.json").read_text())


def _bind_witness(app: Path, submission: dict[str, object]) -> None:
    result = submission["result"]

    def render(value: object) -> str:
        if isinstance(value, dict) and set(value) >= {"numerator", "denominator"}:
            return str(Fraction(value["numerator"], value["denominator"]))
        if isinstance(value, list):
            return ",".join(
                render(item) if isinstance(item, dict) else str(item) for item in value
            )
        return str(value)

    text = (
        "\n".join(
            [
                "apollonius-coefficient-certificate-v1",
                f"multiplier: {render(result['multiplier'])}",
                "circle_coefficients: "
                + ",".join(render(item) for item in result["circle_coefficients"]),
                "distance_coefficients: "
                + ",".join(render(item) for item in result["distance_coefficients"]),
            ]
        )
        + "\n"
    )
    path = app / "evidence/answer.txt"
    path.write_text(text)
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_accepts_alternative_normalization(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    result = sub["result"]
    result.update(
        {
            "k": _q("1/2"),
            "c": _q("4"),
            "p": _q("4/3"),
            "q": _q("-4"),
            "center": _q("-4/3"),
            "radius": _q("8/3"),
            "circle_coefficients": _qs(["1", "1", "8/3", "-16/3"]),
            "distance_coefficients": _qs(["3/4", "3/4", "2", "-4"]),
            "multiplier": _q("3/4"),
        }
    )
    _bind_witness(app, sub)
    _fixtures._write_json(app / "submission.json", sub)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_corrupt_proportionality(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    sub["result"]["distance_coefficients"][2] = _q("23")
    _fixtures._write_json(app / "submission.json", sub)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_extra_result_field_is_protocol_only(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["unexpected"] = True
    _bind_witness(app, submission)
    _fixtures._write_json(app / "submission.json", submission)

    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_rejects_unbound_explanation(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    path = app / "evidence/answer.txt"
    path.write_text("polynomial\n")
    sub["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    _fixtures._write_json(app / "submission.json", sub)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 0.0
    assert reward.reward == 0.0


def test_rejects_replaced_input(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0


def test_rejects_string_and_accepts_unreduced_rationals(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path / "string", TASK, "computed")
    submission = _load(app)
    submission["result"]["k"] = "12/1"
    _bind_witness(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0

    task, app, logs = _fixtures._prepare_case(tmp_path / "unreduced", TASK, "computed")
    submission = _load(app)
    original = Fraction(
        submission["result"]["k"]["numerator"],
        submission["result"]["k"]["denominator"],
    )
    submission["result"]["k"] = {
        "numerator": original.numerator * 2,
        "denominator": original.denominator * 2,
    }
    _bind_witness(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("k", {"numerator": 1, "denominator": 1}),
        ("c", {"numerator": 0, "denominator": 1}),
        ("radius", {"numerator": -1, "denominator": 1}),
    ),
)
def test_declared_rational_constraints_are_protocol_requirements(
    tmp_path: Path, field: str, value: dict[str, int]
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"][field] = value
    _bind_witness(app, submission)
    _fixtures._write_json(app / "submission.json", submission)

    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0


def test_evidence_requires_four_coefficients(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["circle_coefficients"].append(_q("0"))
    _bind_witness(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0
    assert reward.reward == 0.0


def test_evidence_requires_string_multiplier(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["multiplier"] = ["-3"]
    _bind_witness(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0
    assert reward.reward == 0.0


def test_rejects_oversized_evidence(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    path = app / "evidence/answer.txt"
    path.write_text("x" * 1_000_000)
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    _fixtures._write_json(app / "submission.json", submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 0.0
    assert reward.reward == 0.0


def test_result_shape_failure_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["unexpected"] = "ignored by the math check"
    _bind_witness(app, submission)
    _fixtures._write_json(app / "submission.json", submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0
    assert reward.reward == 0.0


@pytest.mark.parametrize(
    ("field", "index"),
    (("multiplier", None), ("circle_coefficients", 0), ("distance_coefficients", 0)),
)
def test_unencodable_rational_fails_closed(
    tmp_path: Path, field: str, index: int | None
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    if index is None:
        submission["result"][field] = "\ud800"
    else:
        submission["result"][field][index] = "\ud800"
    _fixtures._write_json(app / "submission.json", submission)

    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0


def test_evidence_lines_reject_extra_spaces(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    path = app / "evidence/answer.txt"
    path.write_text(" " + path.read_text())
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    _fixtures._write_json(app / "submission.json", submission)

    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0
    assert reward.reward == 0.0
