from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "steiner-triple-system-27"


def test_accepts_alternative_point_relabeling(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    permutation = {point: (5 * point + 7) % 27 for point in range(27)}
    submission["result"]["blocks"] = [
        [permutation[point] for point in block]
        for block in submission["result"]["blocks"]
    ]
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_duplicate_pair(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["blocks"][0] = submission["result"]["blocks"][1]
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_undeclared_witness_key_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"] = [
        {"path": "evidence/answer.txt", "sha256": "sha256:" + "0" * 64}
    ]
    _fixtures._write_json(app / "submission.json", submission)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_relabeling_without_ceremonial_file_is_accepted(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    permutation = {point: (5 * point + 7) % 27 for point in range(27)}
    submission["result"]["blocks"] = [
        [permutation[point] for point in block]
        for block in submission["result"]["blocks"]
    ]
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_input_tamper_is_reported_separately(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    frozen = json.loads((app / "input.json").read_text())
    frozen["source"]["row"] = 999
    _fixtures._write_json(app / "input.json", frozen)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["input_binding"] == 0.0
    assert rejected.reward == 0.0
