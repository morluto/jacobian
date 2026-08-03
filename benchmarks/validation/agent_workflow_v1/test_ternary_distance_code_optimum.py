from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK_NAME = "ternary-distance-code-optimum"
TASK = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "datasets"
    / "agent-workflow-v1"
    / TASK_NAME
)


def _case(
    tmp_path: Path, submission: dict, *, label: str = "case", tamper_input: bool = False
):
    root = tmp_path / label
    app = root / "app"
    logs = root / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    if tamper_input:
        source = json.loads((app / "input.json").read_text())
        source["claimed_optimum"] = 17
        support._write_json(app / "input.json", source)
    evidence = (TASK / "solution" / "answer.txt").read_bytes()
    (app / "evidence" / "answer.txt").write_bytes(evidence)
    submission = deepcopy(submission)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence).hexdigest()
    )
    support._write_json(app / "submission.json", submission)
    return TASK, app, logs


def _submission() -> dict:
    return json.loads((TASK / "solution" / "submission.json").read_text())


def test_reference_certificate_passes(tmp_path: Path) -> None:
    result = support._run_verifier(*_case(tmp_path, _submission(), label="reference"))
    assert result["reward"] == pytest.approx(1.0)
    assert result["false_certification"] is False


def test_alphabet_permutation_is_accepted(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["codewords"] = [
        "".join(str((int(symbol) + 1) % 3) for symbol in word)
        for word in reversed(submission["result"]["codewords"])
    ]
    result = support._run_verifier(*_case(tmp_path, submission, label="alternative"))
    assert result["reward"] == pytest.approx(1.0)


def test_pair_distance_corruption_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["codewords"][1] = "000001"
    result = support._run_verifier(*_case(tmp_path, submission, label="distance"))
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_wrong_dual_multiplier_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["upper_bound_certificate"]["dual_multipliers"]["order_1"] = (
        "1/2"
    )
    result = support._run_verifier(*_case(tmp_path, submission, label="dual"))
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0


def test_noncanonical_rational_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["result"]["upper_bound_certificate"]["dual_multipliers"]["order_2"] = (
        "2/12"
    )
    result = support._run_verifier(*_case(tmp_path, submission, label="fraction"))
    assert result["correctness"] == 0.0


def test_verified_claim_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["claimed_assurance"] = "VERIFIED"
    result = support._run_verifier(*_case(tmp_path, submission, label="verified"))
    assert result["false_certification"] is True
    assert result["reward"] == 0.0


def test_input_tampering_is_rejected(tmp_path: Path) -> None:
    result = support._run_verifier(
        *_case(tmp_path, _submission(), label="input", tamper_input=True)
    )
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0
