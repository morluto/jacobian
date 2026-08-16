from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "radical-distance-triangle-certificate"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})


def test_accepts_exact_geometric_certificate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_invalid_geometric_claims(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["method"] = "NUMERICAL_SAMPLING"
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0

    task, app, logs = _case(tmp_path / "expansion")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["expanded_radicands"][0]["constant"] = 2
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0
