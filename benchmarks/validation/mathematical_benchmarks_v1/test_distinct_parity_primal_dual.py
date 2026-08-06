from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "distinct-parity-primal-dual"


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
    (app / "evidence" / "distinct-parity-certificate.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)


def test_accepts_alternative_optimal_construction(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["even_numbers"] = [*range(2, 50, 2), 56]
    submission["result"]["odd_numbers"] = list(range(1, 74, 2))
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == 1.0


def test_rejects_feasible_but_suboptimal_construction(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["even_numbers"] = [*range(2, 54, 2), 98]
    submission["result"]["odd_numbers"] = list(range(1, 70, 2))
    submission["result"]["objective"] = 5 * 27 + 7 * 35
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_corrupted_upper_bound_frontier(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["frontier"][18]["objective"] = 385
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_accepts_order_independent_frontier(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["frontier"].reverse()
    _rewrite(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == 1.0


def test_rejects_float_objective(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["objective"] = 384.0
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_bool_in_frontier(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["frontier"][0]["odd_count"] = True
    _rewrite(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_oversized_evidence(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "distinct-parity-certificate.json"
    evidence_path.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_decouples_correctness_from_wrong_assurance(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    _rewrite(app, submission)

    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["reward"] == 0.0
