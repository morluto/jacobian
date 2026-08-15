from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "putnam-2adic-induction-audit"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _inject_marker(app: Path, submission: dict) -> None:
    evidence = app / "evidence" / "answer.txt"
    text = evidence.read_text()
    lines = [line for line in text.splitlines() if not line.startswith("RESULT_JSON:")]
    marker = "RESULT_JSON: " + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    lines.append(marker)
    evidence.write_text("\n".join(lines) + "\n")
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)


def _rewrite(app: Path, submission: dict) -> None:
    _inject_marker(app, submission)
    _fixtures._write_json(app / "submission.json", submission)


def test_oracle_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("valuation_induction", "sub_one_term_lower_bounds", 1), [1, 2]),
        (("target_transfer", "b_difference"), [2, 3]),
        (("finite_testing_role",), "FINITE_CASES_PROVE_ALL_K"),
    ],
)
def test_rejects_corrupted_induction_certificates(
    tmp_path: Path,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _rewrite(app, submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_visible_input_tampering(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    (app / "input.json").write_text("{}")
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_witness_without_result_marker_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_witness_result_mismatch_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("RESULT_JSON: {}\n")
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence)
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0
