from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK_NAME = "knight-cycle-game-strategy"
TASK = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "datasets"
    / "agent-workflow-v1"
    / TASK_NAME
)


def _submission() -> dict:
    return json.loads((TASK / "solution" / "submission.json").read_text())


def _case(tmp_path: Path, submission: dict, *, label: str, tamper_input: bool = False):
    root = tmp_path / label
    app = root / "app"
    logs = root / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    if tamper_input:
        source = json.loads((app / "input.json").read_text())
        source["claimed_optimum"] = 99
        support._write_json(app / "input.json", source)
    evidence = (TASK / "solution" / "answer.txt").read_bytes()
    (app / "evidence" / "answer.txt").write_bytes(evidence)
    submission = deepcopy(submission)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence).hexdigest()
    )
    support._write_json(app / "submission.json", submission)
    return TASK, app, logs


def test_reference_strategy_certificate_passes(tmp_path: Path) -> None:
    reward = support._run_verifier(*_case(tmp_path, _submission(), label="reference"))
    assert reward["reward"] == pytest.approx(1.0)
    assert reward["false_certification"] is False


def test_other_checkerboard_parity_and_reordering_pass(tmp_path: Path) -> None:
    submission = _submission()
    lower = submission["result"]["lower_strategy"]
    lower["parity"] = 1
    lower["eligible_sites"] = [
        f"{x},{y}"
        for x in range(20, 0, -1)
        for y in range(20, 0, -1)
        if (x + y) % 2 == 1
    ]
    upper = submission["result"]["upper_strategy"]
    upper["cycles"].reverse()
    for cycle in upper["cycles"]:
        cycle["sites"].reverse()
        cycle["opposite_pairs"].reverse()
        for pair in cycle["opposite_pairs"]:
            pair.reverse()
    reward = support._run_verifier(*_case(tmp_path, submission, label="alternative"))
    assert reward["reward"] == pytest.approx(1.0)


def test_nonpartition_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    cycles = submission["result"]["upper_strategy"]["cycles"]
    cycles[-1] = deepcopy(cycles[0])
    reward = support._run_verifier(*_case(tmp_path, submission, label="partition"))
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0


def test_adjacent_vertices_cannot_be_declared_opposite(tmp_path: Path) -> None:
    submission = _submission()
    cycle = submission["result"]["upper_strategy"]["cycles"][0]
    cycle["opposite_pairs"] = [
        [cycle["sites"][0], cycle["sites"][1]],
        [cycle["sites"][2], cycle["sites"][3]],
    ]
    reward = support._run_verifier(*_case(tmp_path, submission, label="opposite"))
    assert reward["correctness"] == 0.0


def test_invalid_checkerboard_site_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    sites = submission["result"]["lower_strategy"]["eligible_sites"]
    sites[-1] = "1,2"
    reward = support._run_verifier(*_case(tmp_path, submission, label="parity"))
    assert reward["correctness"] == 0.0


def test_verified_claim_is_rejected(tmp_path: Path) -> None:
    submission = _submission()
    submission["claimed_assurance"] = "VERIFIED"
    reward = support._run_verifier(*_case(tmp_path, submission, label="verified"))
    assert reward["false_certification"] is True
    assert reward["reward"] == 0.0


def test_input_tampering_is_rejected(tmp_path: Path) -> None:
    reward = support._run_verifier(
        *_case(tmp_path, _submission(), label="input", tamper_input=True)
    )
    assert reward["correctness"] == 0.0
    assert reward["reward"] == 0.0
