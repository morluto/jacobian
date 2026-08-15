from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _metadata,
    _verifier,
)

TASK = "local-ring-diagonal-similarity-certificate"


def _prepare(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def test_oracle_certificate_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    (app / "input.json").write_text("{}")
    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["input_binding"] == 0.0
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 0.0


def test_integral_float_permutation_is_rejected_without_crashing(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["unit_permutation"] = [
        float(value) for value in submission["result"]["unit_permutation"]
    ]
    _fixtures._write_json(app / "submission.json", submission)

    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_reordered_matched_pairs_are_accepted(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["matched_pairs"].reverse()
    _fixtures._write_json(app / "submission.json", submission)

    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 1.0
    assert reward.reward == 1.0


def test_corrupted_matrix_product_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _prepare(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["PA"][0][0] = 1
    _fixtures._write_json(app / "submission.json", submission)

    reward = _verifier._run_verifier(task, app, logs)
    assert reward.details["correctness"] == 0.0
    assert reward.reward == 0.0


def test_task_declares_input_binding_decoupling() -> None:
    assert _metadata.is_input_binding_decoupled(TASK) is True
