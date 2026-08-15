"""Fixture preparation and task catalog constants.

Owns the curated task-selection registries used by generic verifier tests
plus the canonical submission/witness fixture builder ``_prepare_case``
and its helpers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation.public_reproductions_v1._paths import AGENT_TASKS, TASKS

VERIFIER_TASKS = tuple(
    sorted(
        ref.path.name
        for ref in AGENT_TASKS
        if (ref.path / "tests" / "verifier.py").is_file()
    )
)
SINGLE_EVIDENCE_TASKS = tuple(
    task_name
    for task_name in VERIFIER_TASKS
    if json.loads(
        (TASKS / task_name / "environment" / "submission_schema.json").read_text()
    )["properties"]["witness"].get("maxItems")
    == 1
)


def _task_tree_snapshot() -> dict[str, str]:
    return {
        path.relative_to(TASKS).as_posix(): _digest(path)
        for path in sorted(TASKS.rglob("*"))
        if path.is_file()
    }


def _task(task_name: str) -> Path:
    task = TASKS / task_name
    assert task.is_dir()
    return task


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _bind_result_evidence(app: Path, submission: dict) -> None:
    evidence_path = app / "evidence" / "answer.txt"
    lines = evidence_path.read_text().splitlines()
    marker = "RESULT_JSON: " + json.dumps(
        submission["result"], sort_keys=True, separators=(",", ":")
    )
    boundary = submission["result"].get("boundary_family")
    boundary_marker = (
        "BOUNDARY_FAMILY_JSON: "
        + json.dumps(boundary, sort_keys=True, separators=(",", ":"))
        if boundary is not None
        else None
    )
    evidence_path.write_text(
        "\n".join(
            marker
            if line.startswith("RESULT_JSON:")
            else boundary_marker
            if boundary_marker is not None and line.startswith("BOUNDARY_FAMILY_JSON:")
            else line
            for line in lines
        )
        + "\n"
    )
    submission["witness"][0]["sha256"] = _digest(evidence_path)


def _prepare_case(
    tmp_path: Path,
    task_name: str,
    scenario: str,
) -> tuple[Path, Path, Path]:
    task = _task(task_name)
    root = tmp_path / task_name / scenario
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    submission = json.loads((task / "solution" / "submission.json").read_text())
    for descriptor in submission["witness"]:
        evidence_path = Path(descriptor["path"])
        assert not evidence_path.is_absolute() and ".." not in evidence_path.parts
        destination = app / evidence_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fixture = task / "solution" / evidence_path.name
        if fixture.is_file():
            shutil.copy2(fixture, destination)
        else:
            shutil.copy2(task / "solution" / "answer.txt", destination)
        descriptor["sha256"] = _digest(destination)
    _write_json(app / "submission.json", submission)
    return task, app, logs
