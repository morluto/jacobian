from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "rank-one-spectral-limit-certificate"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _fraction(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _rewrite(app: Path, submission: dict) -> None:
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence/spectral-certificate.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)


def test_accepts_alternative_exact_checkpoints(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    checkpoints = []
    for n in (5, 11, 19, 27):
        total = sum((Fraction(1, k**3 - k) for k in range(2, n + 1)), Fraction())
        checkpoints.append(
            {"n": n, "reciprocal_sum": _fraction(total), "root": _fraction(1 / total)}
        )
    submission["result"]["checkpoints"] = checkpoints
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs).reward == 1.0


def test_accepts_reordered_distinct_checkpoints(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["checkpoints"].reverse()
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs).reward == 1.0


def test_rejects_sampled_but_corrupt_root(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["checkpoints"][1]["root"] = {"numerator": 4, "denominator": 1}
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_duplicate_checkpoint_shortcut(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["checkpoints"][2] = submission["result"]["checkpoints"][1]
    _rewrite(app, submission)
    assert support._run_verifier(task, app, logs).details["correctness"] == 0.0
