from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/reconstruction-deck-certificate"
TASK_ID = "jacobian/reconstruction-deck-certificate"


def case(tmp_path: Path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def _payload(result: object) -> dict[str, object]:
    return {"schema_version": "1", "task_id": TASK_ID, "result": result}


def write(app: Path, submission: dict, *, payload: object | None = None) -> None:
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        json.dumps(
            _payload(submission["result"]) if payload is None else payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")


def run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    result = run(app, logs)
    assert result.reward == result.details["aggregate_reward"] == 1.0


def test_wrong_reconstruction_and_duplicate_card_fail(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    submission["result"]["embeddings"][0]["local_to_original"][:2] = reversed(
        submission["result"]["embeddings"][0]["local_to_original"][:2]
    )
    write(app, submission)
    assert run(app, logs).details["aggregate_reward"] == 0.0

    app, logs, submission = case(tmp_path / "duplicate")
    submission["result"]["embeddings"][-1] = submission["result"]["embeddings"][0]
    write(app, submission)
    assert run(app, logs).details["aggregate_reward"] == 0.0


def test_tampered_input_and_witness_fail_independently(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    (app / "input.json").write_text("{}\n")
    result = run(app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["mathematics"] == 1.0
    assert result.reward == 0.0

    app, logs, _ = case(tmp_path / "witness")
    (app / "evidence/answer.txt").write_text("tampered\n")
    result = run(app, logs)
    assert result.details["witness_validity"] == 0.0
    assert result.details["mathematics"] == 1.0
    assert result.reward == 0.0


def test_witness_binding_is_type_strict(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    payload = _payload(submission["result"])
    payload["result"] = json.loads(json.dumps(payload["result"]))
    payload["result"]["original_edges"][0][1] = True
    write(app, submission, payload=payload)
    result = run(app, logs)
    assert result.details["mathematics"] == 1.0
    assert result.details["witness_validity"] == 0.0
    assert result.reward == 0.0


def test_witness_with_oversized_whitespace_padding_passes(tmp_path: Path) -> None:
    """A digest-bound witness is streamed without an arbitrary byte ceiling."""
    app, logs, submission = case(tmp_path)
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        json.dumps(
            _payload(submission["result"]), sort_keys=True, separators=(",", ":")
        )
        + "\n"
        + " " * (64 * 1024 * 1024 + 424242)
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")
    assert run(app, logs).reward == 1.0


def test_witness_with_large_whitespace_prefix_streams_quickly(tmp_path: Path) -> None:
    """Leading whitespace is discarded while parsing a legal large witness."""
    app, logs, submission = case(tmp_path)
    evidence = app / "evidence/answer.txt"
    evidence.write_text(
        " \n" * (32 * 1024 * 1024)
        + json.dumps(
            _payload(submission["result"]), sort_keys=True, separators=(",", ":")
        )
        + "\n"
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")
    started = time.monotonic()
    assert run(app, logs).reward == 1.0
    assert time.monotonic() - started < 15.0


def test_malformed_submission_and_witness_are_rejected(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    (app / "submission.json").write_text('{"result":{},"result":{}}\n')
    assert run(app, logs).reward == 0.0

    app, logs, submission = case(tmp_path / "witness")
    write(app, submission, payload={**_payload(submission["result"]), "extra": True})
    result = run(app, logs)
    assert result.details["witness_validity"] == 0.0
    assert result.reward == 0.0
