from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation._verifier_child import (
    VerifierExecutionError,
    VerifierOutput,
    run_verifier_in_child,
)

ROOT = Path(__file__).parents[3]
DATASET = ROOT / "benchmarks" / "datasets" / "research-diagnostics-v1"
TASK_EVIDENCE = {
    "jcb-postdoc-004": "counterexample.json",
    "jcb-postdoc-014": "syzygy-certificate.json",
    "jcb-postdoc-015": "finite-core.json",
    "jcb-postdoc-016": "powerful-window.json",
    "jcb-postdoc-019": "nullstellensatz-certificate.json",
}
TASK_EVIDENCE_SCHEMAS = {
    "jcb-postdoc-004": "certificate_schema.json",
    "jcb-postdoc-014": "certificate_schema.json",
    "jcb-postdoc-015": "evidence_schema.json",
    "jcb-postdoc-016": "evidence_schema.json",
    "jcb-postdoc-019": "certificate_schema.json",
}


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def prepare_case(
    tmp_path: Path,
    task_name: str,
) -> tuple[Path, Path, Path]:
    task = DATASET / task_name
    root = tmp_path / task_name
    app = root / "app"
    logs = root / "logs"
    evidence = app / "evidence"
    evidence.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    filename = TASK_EVIDENCE[task_name]
    shutil.copy2(task / "solution" / filename, evidence / filename)
    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = digest(evidence / filename)
    write_json(app / "submission.json", submission)
    return task, app, logs


def run_verifier(task: Path, app: Path, logs: Path) -> VerifierOutput:
    try:
        return run_verifier_in_child(task=task, app=app, logs=logs)
    except (ValueError, VerifierExecutionError):
        return VerifierOutput(
            reward=0.0,
            details={
                "correctness": 0.0,
                "evidence_validity": 0.0,
                "scope_accuracy": 0.0,
                "assurance_calibration": 0.0,
                "limitation_accuracy": 0.0,
                "protocol_compliance": 0.0,
                "false_certification": False,
            },
        )
