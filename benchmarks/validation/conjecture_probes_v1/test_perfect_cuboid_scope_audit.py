from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/perfect-cuboid-scope-audit"


def test_public_instruction_specifies_schema_version_as_json_string() -> None:
    instruction = (TASK / "instruction.md").read_text()
    assert 'Use schema version `1` (the JSON string\n`"1"`)' in instruction


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def _write(app: Path, submission: dict) -> None:
    evidence = app / "evidence/answer.txt"
    payload = {
        "schema_version": "1",
        "task_id": "jacobian/perfect-cuboid-scope-audit",
        "result": submission["result"],
    }
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


def _run(app: Path, logs: Path) -> dict:
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_and_reordered_cases_receive_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["cases"].reverse()
    _write(app, submission)
    reward = _run(app, logs)
    assert reward.details["aggregate_reward"] == 1.0
    assert reward.details["mathematics"] == 1.0


def test_reordered_aligned_face_pairs_receive_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    row = submission["result"]["cases"][0]
    order = (2, 0, 1)
    row["face_radicands"] = [row["face_radicands"][index] for index in order]
    row["face_roots"] = [row["face_roots"][index] for index in order]
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 1.0


def test_euler_brick_cannot_be_promoted_to_perfect_cuboid(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["cases"][0]["class"] = "PERFECT_CUBOID"
    submission["result"]["contains_perfect_cuboid"] = True
    _write(app, submission)
    reward = _run(app, logs)
    assert reward.details["mathematics"] == 0.0
    assert reward.details["aggregate_reward"] == 0.0


def test_wrong_root_omission_and_duplicate_fail_closed(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["cases"][0]["space_root"] = 271
    _write(app, submission)
    assert _run(app, logs).details["mathematics"] == 0.0

    app, logs, submission = _case(tmp_path / "missing")
    submission["result"]["cases"].pop()
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 0.0

    app, logs, submission = _case(tmp_path / "duplicate")
    submission["result"]["cases"][-1] = submission["result"]["cases"][0]
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 0.0


def test_tampered_input_and_evidence_are_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    frozen = json.loads((app / "input.json").read_text())
    frozen["cases"][0]["edges"][0] += 1
    (app / "input.json").write_text(json.dumps(frozen))
    _write(app, submission)
    reward = _run(app, logs)
    assert reward.details["input_binding"] == 0.0
    assert reward.details["mathematics"] == 1.0
    assert reward.details["aggregate_reward"] == 0.0

    app, logs, submission = _case(tmp_path / "evidence")
    _write(app, submission)
    (app / "evidence/answer.txt").write_text("{}\n")
    reward = _run(app, logs)
    assert reward.details["witness_validity"] == 0.0
    assert reward.details["aggregate_reward"] == 0.0


def test_large_declared_witness_remains_valid_without_an_arbitrary_byte_cap(
    tmp_path: Path,
) -> None:
    app, logs, submission = _case(tmp_path)
    evidence = app / "evidence/answer.txt"
    evidence.write_bytes(evidence.read_bytes() + (b" \n" * (2 * 1024 * 1024)))
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission) + "\n")
    reward = _run(app, logs)
    assert reward.details["witness_validity"] == 1.0
    assert reward.details["aggregate_reward"] == 1.0
