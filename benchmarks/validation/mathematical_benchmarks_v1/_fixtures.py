"""Fixture preparation and task catalog constants.

Owns the curated task-selection registries used by generic verifier tests
(verification-record, rational, resource-derived, single-evidence, and the
full verifier-task list) plus the canonical submission/evidence fixture
builder ``_prepare_case`` and its helpers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._paths import AGENT_TASKS, TASKS
from jsonschema import Draft202012Validator

VERIFICATION_RECORD_TASKS = (
    "finite-partition",
    "hermite-normal-form",
    "polynomial-map-collision",
    "polynomial-normalization",
    "sat-witness",
)
RATIONAL_TASK = "rational-linear-solution"
# Curated COMPUTED-ceiling sample for oracle + naked-VERIFIED assurance checks.
# Do not auto-expand to every COMPUTED task; add a name only when that contract
# is the intended coverage and no leaf already owns the smoke.
RESOURCE_DERIVED_TASKS = (
    "elementwise-fixed-no-global-invariant",
    "autoformalization-semantic-audit",
    "calendar-good-days-audit",
    "finite-magma-countermodel",
    "gaussian-moment-generality-audit",
    "generated-lemma-vacuity-audit",
    "inverse-distance-remainder-audit",
    "lagrangian-projection-proof-audit",
    "lcm-highly-abundant-scope-audit",
    "lean-transitive-axiom-audit",
    "log-exponent-recovery",
    "matrix-square-zero-counterexample",
    "metric-tsp-proof-repair",
    "noncompact-lefschetz-proof-audit",
    "parameterized-sharp-bound-audit",
    "polynomial-divisibility-uniqueness",
    "polynomial-tail-counterexample",
    "putnam-2adic-induction-audit",
    "random-function-expectation-audit",
    "research-status-evidence-audit",
    "squarefree-class-independence-audit",
    "subspace-direct-sum-counterexample",
    "well-total-domination-counterexample",
)
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
    )["properties"]["evidence"].get("maxItems")
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
    submission["evidence"][0]["sha256"] = _digest(evidence_path)


def _sat_record(task: Path, app: Path, submission: Mapping[str, object]) -> dict:
    input_data = json.loads((task / "environment" / "input.json").read_text())
    result = submission["result"]
    assert isinstance(result, dict)
    assignment = result["assignment"]
    assert isinstance(assignment, dict)
    key = ",".join("1" if assignment[name] else "0" for name in input_data["variables"])
    authorized = json.loads((task / "tests" / "authorized_records.json").read_text())[
        key
    ]
    return {
        "task_id": input_data["task_id"],
        "input_sha256": _digest(app / "input.json"),
        "conclusion": "TRUE",
        "status": "VERIFIED_SATISFYING",
        "assignment": assignment,
        "scope": submission["scope"],
        "verification_record": authorized,
    }


def _bound_record(task_name: str, task: Path, app: Path, submission: dict) -> dict:
    if task_name == "sat-witness":
        return _sat_record(task, app, submission)
    return json.loads((task / "tests" / "authorized_record.json").read_text())


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
    submission.pop("verification_record_uri", None)
    for descriptor in submission["evidence"]:
        evidence_path = Path(descriptor["path"])
        assert not evidence_path.is_absolute() and ".." not in evidence_path.parts
        destination = app / evidence_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fixture = task / "solution" / evidence_path.name
        if fixture.is_file():
            shutil.copy2(fixture, destination)
        elif evidence_path.suffix == ".json":
            _write_json(
                destination,
                {
                    "schema_version": "1",
                    "task_id": submission["task_id"],
                    "result": submission["result"],
                    "limitations": submission["limitations"],
                },
            )
            descriptor["sha256"] = _digest(destination)
        else:
            shutil.copy2(task / "solution" / "answer.txt", destination)
            descriptor["sha256"] = _digest(destination)

    if scenario == "computed":
        submission["claimed_assurance"] = "COMPUTED"
    else:
        submission["claimed_assurance"] = "VERIFIED"
    if scenario in {"bound", "invalid"}:
        record = (
            _bound_record(task_name, task, app, submission)
            if scenario == "bound"
            else {}
        )
        record_path = app / "evidence" / "verification-record.json"
        _write_json(record_path, record)
        if scenario == "bound":
            schema = json.loads(
                (task / "environment" / "verification_record_schema.json").read_text()
            )
            Draft202012Validator(schema).validate(record)
        submission["verification_record_uri"] = {
            "path": "evidence/verification-record.json",
            "sha256": _digest(record_path),
        }
    _write_json(app / "submission.json", submission)
    return task, app, logs
