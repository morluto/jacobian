from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.public_reproductions_v1 import support

TASK = "superposition-proof-replay"


def _prepare_resolution_replay(tmp_path: Path):
    task, app, logs = support._prepare_case(
        tmp_path, "superposition-proof-replay", "computed"
    )
    certificate = task / "solution" / "resolution-proof.json"
    target = app / "evidence" / "resolution-proof.json"
    target.write_bytes(certificate.read_bytes())
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["evidence"][0]["sha256"] = support._digest(target)
    support._write_json(submission_path, submission)
    return task, app, logs


def test_superposition_replay_accepts_independent_topological_order(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_resolution_replay(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    # Steps 4 and 5 are independent at this point; this is not answer-string replay.
    submission["result"]["steps"] = [
        {"child": 5, "parents": [6, 2]},
        {"child": 8, "parents": [6, 5]},
        {"child": 4, "parents": [8, 3]},
        {"child": 7, "parents": [4, 1]},
    ]
    evidence_path = app / "evidence" / "resolution-proof.json"
    payload = json.loads(evidence_path.read_text())
    payload["result"] = submission["result"]
    support._write_json(evidence_path, payload)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_superposition_replay_rejects_non_resolvent_parent_pair(
    tmp_path: Path,
) -> None:
    task, app, logs = _prepare_resolution_replay(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["steps"][0]["parents"] = [1, 2]
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_superposition_replay_rejects_forward_reference(tmp_path: Path) -> None:
    task, app, logs = _prepare_resolution_replay(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["steps"][0], submission["result"]["steps"][1] = (
        submission["result"]["steps"][1],
        submission["result"]["steps"][0],
    )
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_superposition_replay_rejects_promoted_intermediate_axioms(
    tmp_path: Path,
) -> None:
    """Relabeling intended resolvents as axioms must not bypass the proof."""
    task, app, logs = _prepare_resolution_replay(tmp_path)
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    # Attack: treat clauses 4,5,8 as axioms and only derive the target.
    submission["result"]["axioms"] = [1, 2, 3, 4, 5, 6, 8]
    submission["result"]["steps"] = [{"child": 7, "parents": [1, 4]}]
    evidence_path = app / "evidence" / "resolution-proof.json"
    payload = json.loads(evidence_path.read_text())
    payload["result"] = submission["result"]
    support._write_json(evidence_path, payload)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
