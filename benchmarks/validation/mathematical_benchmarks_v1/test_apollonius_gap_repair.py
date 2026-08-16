from __future__ import annotations

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


def _write(app: Path, submission: dict[str, object]) -> None:
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})


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
    _write(app, sub)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_corrupt_proportionality(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    sub["result"]["distance_coefficients"][2] = _q("23")
    _write(app, sub)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_extra_result_field_is_protocol_only(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["unexpected"] = True
    _write(app, submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
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
    _write(app, submission)
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
    _write(app, submission)
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
    _write(app, submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0


def test_extra_circle_coefficient_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["circle_coefficients"].append(_q("0"))
    _write(app, submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0


def test_list_multiplier_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["multiplier"] = ["-3"]
    _write(app, submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0


def test_unused_evidence_file_does_not_affect_reward(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    (app / "evidence").mkdir(exist_ok=True)
    (app / "evidence/answer.txt").write_text("x" * 1000)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 1.0


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
    _write(app, submission)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.reward == 0.0
