from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite
from benchmarks.validation._verifier_child import (
    VerifierExecutionError,
    run_verifier_in_child,
)
from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[3]
TASKS = ROOT / "benchmarks" / "datasets" / "agent-workflow-v1"
AGENT_TASKS = get_suite("agent-workflow-v1").tasks
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
# Tasks whose verifier reports scope independently of assurance typing.
SCOPE_INDEPENDENT_ASSURANCE_TASKS = (
    "prime-power-divisibility-gap-audit",
    "sine-integral-asymptotic-audit",
    "steiner-triple-system-27",
)
# Tasks whose verifier reports mathematical correctness independently of
# workspace input binding, emitting a separate ``input_binding`` diagnostic
# and gating only aggregate reward on both.
INPUT_BINDING_DECOUPLED_TASKS = (
    "extremal-subset-sum-semantic-audit",
    "integer-perturbation-domain-audit",
    "monotone-inverse-continuity-audit",
    "necklace-burnside-certificate",
    "prime-power-divisibility-gap-audit",
    "pythagorean-generator-recurrence",
    "sine-integral-asymptotic-audit",
    "steiner-triple-system-27",
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


def _run_verifier(task: Path, app: Path, logs: Path) -> dict:
    try:
        return run_verifier_in_child(task=task, app=app, logs=logs)
    except (ValueError, VerifierExecutionError):
        return {
            "assurance_calibration": 0.0,
            "correctness": 0.0,
            "evidence_validity": 0.0,
            "false_certification": False,
            "input_binding": 0.0,
            "input_integrity": 0.0,
            "limitation_accuracy": 0.0,
            "protocol": 0.0,
            "protocol_compliance": 0.0,
            "reward": 0.0,
            "scope_accuracy": 0.0,
        }


def _prepare_case(
    tmp_path: Path,
    task_name: str,
    scenario: str,
) -> tuple[Path, Path, Path]:
    task = _task(task_name)
    root = tmp_path / task_name / scenario
    app = root / "app"
    logs = root / "logs"
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(task / "environment" / "input.json", app / "input.json")
    shutil.copy2(task / "solution" / "answer.txt", app / "evidence" / "answer.txt")
    submission = json.loads((task / "solution" / "submission.json").read_text())
    submission.pop("verification_record_uri", None)

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
