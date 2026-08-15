from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "cyclic-polynomial-sum-audit"
TASK_PATH = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/cyclic-polynomial-sum-audit"
)
WITNESS_PATH = "evidence/cyclic-elimination-certificate.json"


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _bind_witness_result(app: Path, submission: dict) -> None:
    """Rebind the result field in the witness JSON to the submission result."""
    witness_file = app / WITNESS_PATH
    payload = json.loads(witness_file.read_text())
    payload["result"] = submission["result"]
    _write_json(witness_file, payload)
    submission["witness"][0]["sha256"] = _digest(witness_file)


def _case(tmp_path: Path):
    root = tmp_path / TASK / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK_PATH / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK_PATH / "solution" / "submission.json").read_text())
    for descriptor in submission["witness"]:
        evidence_path = Path(descriptor["path"])
        destination = app / evidence_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TASK_PATH / "solution" / evidence_path.name, destination)
        descriptor["sha256"] = _digest(destination)
    _write_json(app / "submission.json", submission)
    return TASK_PATH, app, logs


def test_oracle_replays_complete_elimination_certificate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("necessary_polynomial", 2), -10),
        (("proposed_evaluations", 1), "0"),
        (("excluded_branch", "product"), "-111/8"),
        (("excluded_branch", "residual"), "0"),
    ],
)
def test_rejects_corrupted_algebraic_certificates(
    tmp_path: Path, path: tuple[object, ...], replacement: object
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    target = submission["result"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _bind_witness_result(app, submission)
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_keyword_filler_witness(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    witness_path = app / WITNESS_PATH
    witness_path.write_text(
        "pairwise distinct a+b product residual -3/2 " * 64,
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"][0]["sha256"] = _digest(witness_path)
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_rejects_witness_result_not_bound_to_submission(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    witness_path = app / WITNESS_PATH
    evidence = json.loads(witness_path.read_text())
    evidence["result"]["excluded_branch"]["residual"] = "0"
    _write_json(witness_path, evidence)
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"][0]["sha256"] = _digest(witness_path)
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_rejects_oversized_witness_before_parsing(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    witness_path = app / WITNESS_PATH
    witness_path.write_text(" " * (64 * 1024 + 1))
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"][0]["sha256"] = _digest(witness_path)
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0
