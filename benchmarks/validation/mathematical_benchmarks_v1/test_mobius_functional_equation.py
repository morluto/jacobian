from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "mobius-functional-equation"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})


def test_accepts_exact_orbit(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_corrupted_orbit_and_singular_matrix(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["solution_values"][1]["numerator"][0] += 1
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0
    task, app, logs = _case(tmp_path / "matrix")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["coefficient_matrix"][2] = [0, 1, 1]
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_undeclared_witness_key_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"] = [
        {"path": "evidence/answer.txt", "sha256": "sha256:" + "0" * 64}
    ]
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
