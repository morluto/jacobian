from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "fractional-ratio-proof-repair"


def _run(tmp_path: Path, mutate=None):
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    if mutate:
        mutate(submission)
        _fixtures._write_json(app / "submission.json", submission)
    return _verifier._run_verifier(task, app, logs)


def test_oracle_passes(tmp_path: Path) -> None:
    assert _run(tmp_path).reward == 1.0


def test_rejects_wrong_or_malformed_certificate(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path,
            lambda submission: submission["result"]["item_residuals"][7].update(
                value=0
            ),
        ).reward
        == 0.0
    )


def test_rejects_boolean_numeric_field(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path,
            lambda submission: submission["result"].update(constant_residual=False),
        ).reward
        == 0.0
    )


def test_rejects_extra_legacy_or_witness_fields(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path,
            lambda submission: submission.update(
                limitations=["FROZEN_BINARY_RATIO_INSTANCE_ONLY"]
            ),
        ).reward
        == 0.0
    )


def test_rejects_redundant_witness_field(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path,
            lambda submission: submission.update(
                witness=[
                    {"path": "evidence/answer.txt", "sha256": "sha256:" + "0" * 64}
                ]
            ),
        ).reward
        == 0.0
    )


def test_visible_input_tamper_is_a_hard_gate(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "input-tamper")
    (app / "input.json").write_text("{}")
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0
