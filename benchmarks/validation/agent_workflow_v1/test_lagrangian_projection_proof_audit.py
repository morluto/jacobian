from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "lagrangian-projection-proof-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def test_accepts_alternative_coefficients(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result.update(
        P=[["2", "0"], ["0", "2"]],
        W=[["2", "2/5"], ["0", "2"], ["-1", "1"], ["0", "-4/5"]],
        naive_P=[["2", "2/5"], ["-2/5", "2"]],
        naive_Q=[["1", "-1"], ["1", "1"]],
        corrected_first_projection=[["2", "2/5"], ["-2/5", "2"]],
        corrected_second_projection=[["1", "-1"], ["1", "1"]],
    )
    (app / "evidence" / "answer.txt").write_text(
        "The nonzero Lagrangian defect mixes the two naive projections; "
        "the corrected coupled identities reconstruct the exact witness "
        "with scaled coefficients P=2I and Q=I.\n"
        "RESULT_JSON: {}\n"
    )
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_tampered_frozen_input(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    (app / "input.json").write_text(
        '{"frozen_claim":{"standard_symplectic_matrix":[["0","0","1","0"],'
        '["0","0","0","1"],["0","0","0","0"],["0","0","0","0"]]}}'
    )
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_extra_result_field(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["extra_field"] = "malicious"
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_multiple_evidence_descriptors(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"] = [
        submission["evidence"][0],
        submission["evidence"][0],
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0


def test_rejects_corrupted_lagrangian_defect(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["lagrangian_defect"][0][1] = "0"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_oversized_evidence(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    evidence_path = app / "evidence" / "answer.txt"
    submission = json.loads((app / "submission.json").read_text())
    evidence_path.write_text("x" * 2_097_152)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
