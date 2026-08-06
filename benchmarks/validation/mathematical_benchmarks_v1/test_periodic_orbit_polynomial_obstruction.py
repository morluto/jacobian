from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "periodic-orbit-polynomial-obstruction"


def _prepare_periodic_orbit_case(tmp_path: Path):
    task, app, logs = support._prepare_case(
        tmp_path, "periodic-orbit-polynomial-obstruction", "computed"
    )
    source = task / "solution" / "periodic-orbit-certificate.json"
    target = app / "evidence" / "periodic-orbit-certificate.json"
    target.write_bytes(source.read_bytes())
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)
    return task, app, logs


def _bind_periodic_orbit_evidence(app: Path, submission: dict) -> None:
    evidence_path = app / "evidence" / "periodic-orbit-certificate.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["result"] = submission["result"]
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)


def test_periodic_orbit_obstruction_accepts_reordered_prime_reductions(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["modular_reductions"].reverse()
    _bind_periodic_orbit_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_periodic_orbit_obstruction_rejects_one_sided_reduction(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["modular_reductions"] = submission["result"][
        "modular_reductions"
    ][:1]
    _bind_periodic_orbit_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_periodic_orbit_obstruction_rejects_corrupted_mobius_coefficient(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["exact_period_coefficients"] = [1, -1, 1, -1]
    _bind_periodic_orbit_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_periodic_orbit_obstruction_accepts_sign_equivalent_residue_vectors(
    tmp_path: Path,
) -> None:
    """Divisibility is invariant under multiplication by -1, so [1, -1] is
    a valid sign-equivalent alternative to [-1, 1] for residue coefficients.
    """
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    for reduction in submission["result"]["modular_reductions"]:
        reduction["residue_coefficients"] = [1, -1]
    _bind_periodic_orbit_evidence(app, submission)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_periodic_orbit_obstruction_rejects_boolean_coefficients(
    tmp_path: Path,
) -> None:
    """Boolean values must not spoof integer coefficients in any array."""
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["exact_period_coefficients"] = [True, -1, -1, True]
    _bind_periodic_orbit_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_periodic_orbit_obstruction_rejects_boolean_residue_coefficients(
    tmp_path: Path,
) -> None:
    """Boolean values in residue coefficient arrays must also be rejected."""
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["modular_reductions"][0]["residue_coefficients"] = [True, -1]
    _bind_periodic_orbit_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_periodic_orbit_obstruction_rejects_oversized_submission(
    tmp_path: Path,
) -> None:
    """An oversized submission.json must be rejected before parsing to avoid
    OOM or timeout in the memory-limited verifier container.
    """
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission_path.write_text("[" * 2_000_000 + "1" + "]" * 2_000_000)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_periodic_orbit_obstruction_schema_exposes_finite_proof_step_ids() -> None:
    """The agent-visible schema must not reveal the exact proof step as a
    single const, but should expose a finite enum so the agent knows the
    expected identifier format.
    """
    task = support._task("periodic-orbit-polynomial-obstruction")
    schema = json.loads((task / "environment" / "submission_schema.json").read_text())
    result_props = schema["properties"]["result"]["properties"]
    for key in ("infinite_prime_step", "polynomial_identity_step"):
        assert "const" not in result_props[key]
        assert "enum" in result_props[key]
        assert len(result_props[key]["enum"]) > 1


def test_periodic_orbit_obstruction_rejects_checked_above_ceiling(
    tmp_path: Path,
) -> None:
    """CHECKED is above the task's COMPUTED assurance ceiling, so it is an
    unsupported certification that forces reward to zero.
    """
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "CHECKED"
    _bind_periodic_orbit_evidence(app, submission)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_periodic_orbit_obstruction_rejects_oversized_workspace_input(
    tmp_path: Path,
) -> None:
    """An oversized /app/input.json must be rejected before reading to avoid
    OOM in the memory-limited verifier container.
    """
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    (app / "input.json").write_text("x" * 2_000_000)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_periodic_orbit_obstruction_rejects_oversized_evidence(
    tmp_path: Path,
) -> None:
    """An oversized evidence file must be rejected before parsing."""
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    evidence_path = app / "evidence" / "periodic-orbit-certificate.json"
    evidence_path.write_text("x" * 2_000_000)
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_periodic_orbit_obstruction_rejects_boolean_in_evidence_copy(
    tmp_path: Path,
) -> None:
    """When the evidence certificate replaces an integer with boolean ``true``,
    Python equality treats them as equal but the evidence does not exactly copy
    the result.  The verifier must reject this.
    """
    task, app, logs = _prepare_periodic_orbit_case(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    evidence_path = app / "evidence" / "periodic-orbit-certificate.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["result"] = json.loads(
        json.dumps(submission["result"], separators=(",", ":")).replace(
            '"exact_period_coefficients":[1,-1,-1,1]',
            '"exact_period_coefficients":[true,-1,-1,1]',
        )
    )
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0
