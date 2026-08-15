from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "mobius-functional-equation"
TASK_ID = f"jacobian/{TASK}"


def _case(tmp_path: Path):
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    return task, app, logs


def _rewrite(app: Path, submission: dict, *, payload: object | None = None) -> None:
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": submission["result"],
    }
    raw = json.dumps(
        evidence if payload is None else payload, separators=(",", ":")
    ).encode()
    path = app / "evidence" / "functional-equation-certificate.json"
    path.write_bytes(raw)
    submission["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    _fixtures._write_json(app / "submission.json", submission)


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


def test_witness_binds_result_without_limitations(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": {**submission["result"], "matrix_determinant": 3},
    }
    _rewrite(app, submission, payload=payload)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0
    assert result.reward == 0.0
