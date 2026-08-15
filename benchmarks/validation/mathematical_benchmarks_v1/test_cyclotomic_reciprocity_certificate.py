from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

TASK = "cyclotomic-reciprocity-certificate"
TASK_PATH = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/cyclotomic-reciprocity-certificate"
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _inject_result_json(app: Path, submission: dict) -> None:
    """Append a RESULT_JSON marker binding the witness to the submission result."""
    evidence_path = app / "evidence" / "answer.txt"
    text = evidence_path.read_text()
    lines = [line for line in text.splitlines() if not line.startswith("RESULT_JSON:")]
    lines.append(
        "RESULT_JSON:"
        + json.dumps(submission["result"], sort_keys=True, separators=(",", ":"))
    )
    evidence_path.write_text("\n".join(lines) + "\n")
    submission["witness"][0]["sha256"] = _digest(evidence_path)


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
    _inject_result_json(app, submission)
    _write_json(app / "submission.json", submission)
    return TASK_PATH, app, logs


def _rewrite(app: Path, submission: dict) -> None:
    _inject_result_json(app, submission)
    _write_json(app / "submission.json", submission)


def test_rejects_missing_factor(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["factors"].pop()
    _rewrite(app, submission)
    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_corrupted_multiplicity(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["factors"][0]["multiplicity"] = 1
    _rewrite(app, submission)
    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
