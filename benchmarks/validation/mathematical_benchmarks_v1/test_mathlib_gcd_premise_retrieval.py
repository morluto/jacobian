from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "mathlib-gcd-premise-retrieval"


def _case(tmp_path: Path) -> tuple[Path, Path, Path]:
    return _fixtures._prepare_case(tmp_path, TASK, "formal-replay")


def _replace_result(app: Path, result: object) -> None:
    _fixtures._write_json(app / "submission.json", {"result": result})


def test_exact_declaration_application_elaborates(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)

    accepted = _verifier._run_verifier(task, app, logs)

    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_wrong_declaration_is_rejected_by_lean(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _replace_result(
        app,
        {"theorem": "Nat.gcd_zero_left", "arguments": ["n"]},
    )

    rejected = _verifier._run_verifier(task, app, logs)

    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_missing_argument_is_rejected_by_lean(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _replace_result(app, {"theorem": "Nat.gcd_zero_right", "arguments": []})

    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_executable_text_is_rejected_before_elaboration(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    _replace_result(
        app,
        {"theorem": "Nat.gcd_zero_right;#eval", "arguments": ["n"]},
    )

    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_extra_submission_field_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text(encoding="utf-8"))
    submission["tool_trace"] = ["math.find", "math.run"]
    _fixtures._write_json(app / "submission.json", submission)

    assert _verifier._run_verifier(task, app, logs).reward == 0.0
