from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.validation.agent_workflow_v1 import support

TASK = "steiner-triple-system-27"


def test_accepts_alternative_point_relabeling(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    permutation = {point: (5 * point + 7) % 27 for point in range(27)}
    submission["result"]["blocks"] = [
        [permutation[point] for point in block]
        for block in submission["result"]["blocks"]
    ]
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_rejects_duplicate_pair(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["blocks"][0] = submission["result"]["blocks"][1]
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_duplicate_evidence_descriptor(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"].append(dict(submission["evidence"][0]))
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["correctness"] == 1.0
    assert rejected["reward"] == 0.0


def _bind_evidence(app: Path, submission: dict[str, object]) -> None:
    result = submission["result"]
    assert isinstance(result, dict)
    blocks = result["blocks"]
    assert isinstance(blocks, list)
    digest = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    answer = "\n".join(
        [
            "steiner-triple-system-certificate-v1",
            f"result_sha256: {digest}",
            f"order: {result['order']}",
            f"block_count: {len(blocks)}",
            "",
        ]
    )
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(answer)
    evidence = submission["evidence"]
    assert isinstance(evidence, list) and isinstance(evidence[0], dict)
    evidence[0]["sha256"] = f"sha256:{hashlib.sha256(answer.encode()).hexdigest()}"


def test_rejects_checked_assurance_without_partial_reward(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "CHECKED"
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["assurance_calibration"] == 0.0
    assert rejected["protocol_compliance"] == 1.0
    assert rejected["reward"] == 0.0


def test_unverified_assurance_is_protocol_valid_but_unscoreable(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["assurance_calibration"] == 0.0
    assert rejected["protocol_compliance"] == 1.0
    assert rejected["reward"] == 0.0


def test_result_shape_drift_is_protocol_only(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["unexpected"] = True
    _bind_evidence(app, submission)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 1.0
    assert rejected["protocol_compliance"] == 0.0
    assert rejected["reward"] == 0.0


def test_rejects_stale_evidence_after_relabeling(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    permutation = {point: (5 * point + 7) % 27 for point in range(27)}
    submission["result"]["blocks"] = [
        [permutation[point] for point in block]
        for block in submission["result"]["blocks"]
    ]
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0


def test_input_tamper_is_reported_separately(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    frozen = json.loads((app / "input.json").read_text())
    frozen["source"]["row"] = 999
    support._write_json(app / "input.json", frozen)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 1.0
    assert rejected["input_binding"] == 0.0
    assert rejected["reward"] == 0.0


def test_accepts_large_digest_bound_evidence_without_losing_math_diagnostic(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(evidence_path.read_text() + "\n" * (8 * 1024))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_accepts_evidence_without_trailing_newline(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(evidence_path.read_text().rstrip("\n"))
    submission["evidence"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    support._write_json(app / "submission.json", submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["evidence_validity"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)
