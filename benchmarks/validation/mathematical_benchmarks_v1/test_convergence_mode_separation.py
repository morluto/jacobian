from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

_TASK = "convergence-mode-separation"
TASK = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/convergence-mode-separation"
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _bind_result_witness(app: Path, submission: dict) -> None:
    """Rebind the RESULT_JSON marker in the witness file to the submission result."""
    evidence_path = app / "evidence" / "answer.txt"
    lines = evidence_path.read_text().splitlines()
    marker = "RESULT_JSON: " + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    evidence_path.write_text(
        "\n".join(marker if line.startswith("RESULT_JSON:") else line for line in lines)
        + "\n"
    )
    submission["witness"][0]["sha256"] = _digest(evidence_path)


def _case(tmp_path: Path):
    root = tmp_path / _TASK / "computed"
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
        shutil.copy2(TASK / "solution" / evidence_path.name, destination)
        descriptor["sha256"] = _digest(destination)
    _write_json(app / "submission.json", submission)
    return TASK, app, logs


def test_rejects_unbounded_research_status_fact(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["research_scope"]["underlying_problem"] = "ADJUDICATED"
    _bind_result_witness(app, submission)
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == pytest.approx(0.0)


def test_result_marker_carries_semantics_without_keyword_inference(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    (app / "evidence" / "answer.txt").write_text(
        "Evidence is bound below.\nRESULT_JSON: {}\n"
    )
    _bind_result_witness(app, submission)
    _write_json(app / "submission.json", submission)

    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_result_requires_checked_structural_convergence_arguments(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["pointwise_argument"] = {
        "hit_count_per_level": 1,
        "miss_count_per_level": "UNSPECIFIED",
    }
    _bind_result_witness(app, submission)
    _write_json(app / "submission.json", submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == pytest.approx(0.0)
