from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "steiner-triple-system-27"


def test_accepts_alternative_point_relabeling(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    permutation = {point: (5 * point + 7) % 27 for point in range(27)}
    submission["result"]["blocks"] = [
        [permutation[point] for point in block]
        for block in submission["result"]["blocks"]
    ]
    _bind_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_duplicate_pair(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["blocks"][0] = submission["result"]["blocks"][1]
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_duplicate_evidence_descriptor(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["witness"].append(dict(submission["witness"][0]))
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0


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
    evidence = submission["witness"]
    assert isinstance(evidence, list) and isinstance(evidence[0], dict)
    evidence[0]["sha256"] = f"sha256:{hashlib.sha256(answer.encode()).hexdigest()}"


def test_rejects_stale_evidence_after_relabeling(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    permutation = {point: (5 * point + 7) % 27 for point in range(27)}
    submission["result"]["blocks"] = [
        [permutation[point] for point in block]
        for block in submission["result"]["blocks"]
    ]
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_input_tamper_is_reported_separately(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    frozen = json.loads((app / "input.json").read_text())
    frozen["source"]["row"] = 999
    _fixtures._write_json(app / "input.json", frozen)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["input_binding"] == 0.0
    assert rejected.reward == 0.0


def test_accepts_large_digest_bound_evidence_without_losing_math_diagnostic(
    tmp_path: Path,
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(evidence_path.read_text() + "\n" * (8 * 1024))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    _fixtures._write_json(app / "submission.json", submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_accepts_evidence_without_trailing_newline(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(evidence_path.read_text().rstrip("\n"))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    _fixtures._write_json(app / "submission.json", submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0
    assert accepted.reward == pytest.approx(1.0)
