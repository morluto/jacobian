from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "rational-pole-vieta-audit"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})


def test_accepts_exact_symbolic_repair(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_corrupted_polynomial_certificate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["root_sum"] = {"numerator": 5, "denominator": 1005}
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
