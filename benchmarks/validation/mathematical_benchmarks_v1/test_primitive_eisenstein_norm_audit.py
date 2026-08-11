from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "primitive-eisenstein-norm-audit"
LIMITATIONS = [
    "CUBIC_FORM_INTERSECTION_COUNT_NOT_ASSESSED",
    "GENERAL_CRITERION_NOT_PROOF_ASSISTANT_VERIFIED",
    "GENERAL_CRITERION_ONLY_LOCALLY_DEMONSTRATED",
]


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
    (app / "evidence/local-audit.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)


def _base_submission(app: Path) -> dict:
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = list(LIMITATIONS)
    return submission


def test_accepts_alternative_ramified_witness_and_inert_prime(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    submission["result"]["ramified_witness"] = {
        "x": 1,
        "y": 7,
        "norm": 57,
        "gcd": 1,
        "v3": 1,
    }
    submission["result"]["inert_obstruction"] = {
        "prime": 11,
        "zero_pairs": [{"x": 0, "y": 0}],
        "square_primitive_status": "IMPOSSIBLE",
    }
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.reward == 1.0, accepted


def test_rejects_corrupt_residue_certificate(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    submission["result"]["inert_obstruction"]["zero_pairs"].append({"x": 1, "y": 1})
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_nonprimitive_ramified_example(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    submission["result"]["ramified_witness"] = {
        "x": 3,
        "y": 3,
        "norm": 27,
        "gcd": 1,
        "v3": 3,
    }
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0


def test_rejects_boolean_in_integer_certificate_fields(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    submission["result"]["ramified_witness"]["gcd"] = True
    submission["result"]["ramified_witness"]["v3"] = True
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_in_residue_pair_coordinates(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    submission["result"]["inert_obstruction"]["zero_pairs"] = [{"x": False, "y": False}]
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_accepts_equivalent_residue_representatives(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    submission["result"]["inert_obstruction"]["prime"] = 5
    submission["result"]["inert_obstruction"]["zero_pairs"] = [{"x": 5, "y": 5}]
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0, accepted
    assert accepted.reward == 1.0, accepted


def test_separates_protocol_validity_from_mathematical_correctness(
    tmp_path: Path,
) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    submission["claimed_assurance"] = "VERIFIED"
    _rewrite(app, submission)
    result = support._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0, result
    assert result.reward == 0.0, result
    assert result.details["false_certification"] is True


def test_rejects_result_only_evidence_envelope(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    raw = json.dumps(submission["result"], separators=(",", ":")).encode()
    (app / "evidence/local-audit.json").write_bytes(raw)
    submission["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_oversized_evidence_deterministically(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = _base_submission(app)
    oversized = b"x" * (17 * 1024 * 1024)
    (app / "evidence/local-audit.json").write_bytes(oversized)
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(oversized).hexdigest()
    )
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


def test_emits_reward_json_for_malformed_submission(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    (app / "submission.json").write_text("not json")
    result = support._run_verifier(task, app, logs)
    assert result.reward == 0.0
    assert result.details["correctness"] == 0.0
