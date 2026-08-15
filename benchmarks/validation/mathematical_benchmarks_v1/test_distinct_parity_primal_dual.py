from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/distinct-parity-primal-dual"
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _case(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "distinct-parity-primal-dual" / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK / "solution" / "submission.json").read_text())
    for descriptor in submission["witness"]:
        evidence_path = Path(descriptor["path"])
        destination = app / evidence_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fixture = TASK / "solution" / evidence_path.name
        if fixture.is_file():
            shutil.copy2(fixture, destination)
        else:
            _write_json(
                destination,
                {
                    "schema_version": "1",
                    "task_id": "jacobian/distinct-parity-primal-dual",
                    "result": submission["result"],
                },
            )
        descriptor["sha256"] = _digest(destination)
    _write_json(app / "submission.json", submission)
    return TASK, app, logs


def _rewrite(app: Path, submission: dict) -> None:
    evidence = {
        "schema_version": "1",
        "task_id": "jacobian/distinct-parity-primal-dual",
        "result": submission["result"],
    }
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence" / "distinct-parity-certificate.json").write_bytes(raw)
    submission["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    _write_json(app / "submission.json", submission)


def test_accepts_alternative_optimal_construction(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["even_numbers"] = [*range(2, 50, 2), 56]
    submission["result"]["odd_numbers"] = list(range(1, 74, 2))
    _rewrite(app, submission)

    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_rejects_feasible_but_suboptimal_construction(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["even_numbers"] = [*range(2, 54, 2), 98]
    submission["result"]["odd_numbers"] = list(range(1, 70, 2))
    submission["result"]["objective"] = 5 * 27 + 7 * 35
    _rewrite(app, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_corrupted_upper_bound_frontier(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["frontier"][18]["objective"] = 385
    _rewrite(app, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_accepts_order_independent_frontier(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["frontier"].reverse()
    _rewrite(app, submission)

    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_rejects_float_objective(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["objective"] = 384.0
    _rewrite(app, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_bool_in_frontier(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["frontier"][0]["odd_count"] = True
    _rewrite(app, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_oversized_witness(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "distinct-parity-certificate.json"
    evidence_path.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_witness_result_must_match_submission_result(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = {
        "schema_version": "1",
        "task_id": "jacobian/distinct-parity-primal-dual",
        "result": {**submission["result"], "objective": 999},
    }
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    (app / "evidence" / "distinct-parity-certificate.json").write_bytes(raw)
    submission["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_rejects_int_float_witness_mismatch(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence = {
        "schema_version": "1",
        "task_id": "jacobian/distinct-parity-primal-dual",
        "result": json.loads(json.dumps(submission["result"])),
    }
    evidence["result"]["objective"] = 384.0
    raw = json.dumps(evidence, separators=(",", ":")).encode()
    path = app / "evidence" / "distinct-parity-certificate.json"
    path.write_bytes(raw)
    submission["witness"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    _write_json(app / "submission.json", submission)
    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0
