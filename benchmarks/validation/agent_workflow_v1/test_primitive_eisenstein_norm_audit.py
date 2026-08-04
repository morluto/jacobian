from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "primitive-eisenstein-norm-audit"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence/local-audit.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)


def test_accepts_alternative_ramified_witness_and_inert_prime(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["ramified_witness"] = {
        "x": 1,
        "y": 7,
        "norm": 57,
        "gcd": 1,
        "v3": 1,
    }
    submission["result"]["inert_obstruction"] = {
        "prime": 11,
        "zero_pairs": [{"x": 0, "y": 0}],
        "square_primitive_status": "IMPOSSIBLE",
    }
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted["reward"] == 1.0, accepted


def test_rejects_corrupt_residue_certificate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["inert_obstruction"]["zero_pairs"].append({"x": 1, "y": 1})
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_nonprimitive_ramified_example(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["ramified_witness"] = {
        "x": 3,
        "y": 3,
        "norm": 27,
        "gcd": 1,
        "v3": 3,
    }
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
