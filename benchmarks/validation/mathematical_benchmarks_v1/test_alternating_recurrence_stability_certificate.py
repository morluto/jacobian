from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "alternating-recurrence-stability-certificate"
TASK_ID = "jacobian/alternating-recurrence-stability-certificate"


def _rewrite(app: Path, submission: dict) -> None:
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": submission["result"],
    }
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence/stability-certificate.json").write_bytes(raw)
    submission["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    _fixtures._write_json(app / "submission.json", submission)


def test_accepts_alternative_checkpoints(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["checkpoints"] = [
        {
            "n": n,
            "a_n": {"numerator": 2**n, "denominator": 9},
            "difference": {"numerator": 2**n, "denominator": 9},
        }
        for n in (2, 7, 13, 23, 29)
    ]
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_accepts_unordered_equivalent_checkpoints(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["checkpoints"].reverse()
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_finite_simulation_with_wrong_parity_argument(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["negative_delta_bad_parity"] = "EVEN"
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).details["correctness"] == 0.0


def test_rejects_corrupt_closed_form_checkpoint(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["checkpoints"][2]["difference"] = {
        "numerator": 1,
        "denominator": 1,
    }
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_rejects_float_integer_certificate_fields(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["homogeneous_base"] = -7.0
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_rejects_recursive_evidence_without_crashing(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    nested: list = []
    for _ in range(600):
        nested = [nested]
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": nested,
    }
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence/stability-certificate.json").write_bytes(raw)
    submission["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    _fixtures._write_json(app / "submission.json", submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.reward == 0.0
