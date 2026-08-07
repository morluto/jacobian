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


def case(tmp_path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    subprocess.run(
        [sys.executable, str(TASK / "solution/solve.py"), "--root", str(app)],
        check=True,
    )
    return app, logs, json.loads((app / "submission.json").read_text())


def write(app, s):
    payload = {
        "schema_version": "1",
        "task_id": s["task_id"],
        "result": s["result"],
        "limitations": s["limitations"],
    }
    e = app / "evidence/answer.txt"
    e.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")


def run(app, logs):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path):
    app, logs, _ = case(tmp_path)
    assert run(app, logs)["aggregate_reward"] == 1.0


def test_wrong_mapping_and_deleted_vertex_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["embeddings"][0]["local_to_original"][:2] = reversed(
        s["result"]["embeddings"][0]["local_to_original"][:2]
    )
    write(app, s)
    assert run(app, logs)["mathematics"] == 0.0
    app, logs, s = case(tmp_path / "d")
    s["result"]["embeddings"][0]["deleted_vertex"] = 1
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 0.0


def test_corrupt_original_and_duplicate_card_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["result"]["original_edges"][0] = [0, 4]
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 0.0
    app, logs, s = case(tmp_path / "dup")
    s["result"]["embeddings"][-1] = s["result"]["embeddings"][0]
    write(app, s)
    assert run(app, logs)["aggregate_reward"] == 0.0


def test_false_verified_and_tampered_input_fail(tmp_path):
    app, logs, s = case(tmp_path)
    s["claimed_assurance"] = "VERIFIED"
    write(app, s)
    assert run(app, logs)["false_certification"] is True
    app, logs, _ = case(tmp_path / "input")
    (app / "input.json").write_text("{}\n")
    r = run(app, logs)
    assert (
        r["input_binding"] == 0.0
        and r["mathematics"] == 1.0
        and r["aggregate_reward"] == 0.0
    )


def test_unhashable_claimed_assurance_keeps_other_diagnostics(tmp_path):
    """A JSON array/object assurance must not crash the membership test."""
    for value in (["CHECKED", "COMPUTED"], {"level": "CHECKED"}):
        app, logs, s = case(tmp_path / type(value).__name__)
        s["claimed_assurance"] = value
        write(app, s)
        r = run(app, logs)
        assert (
            r["assurance"] == 0.0
            and r["input_binding"] == 1.0
            and r["mathematics"] == 1.0
            and r["evidence"] == 1.0
            and r["scope"] == 1.0
            and r["aggregate_reward"] == 0.0
        )


def test_evidence_with_oversized_whitespace_padding_passes(tmp_path):
    """Legal trailing whitespace beyond any internal ceiling must parse."""
    app, logs, s = case(tmp_path)
    e = app / "evidence/answer.txt"
    answer = (
        json.dumps(
            {
                "schema_version": "1",
                "task_id": s["task_id"],
                "result": s["result"],
                "limitations": s["limitations"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        + " " * (64 * 1024 * 1024 + 424242)
    )
    e.write_text(answer)
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")
    r = run(app, logs)
    assert r["evidence"] == 1.0 and r["aggregate_reward"] == 1.0


def test_evidence_with_large_whitespace_prefix_passes_quickly(tmp_path):
    """A large legal whitespace prefix before the value must not be retained.

    The streaming parser discards the consumed leading-whitespace prefix.
    Without that, a digest-correct file would retain (and repeatedly copy)
    the whole prefix, turning the linear parse into a quadratic one that can
    exhaust the 1 GiB verifier memory limit or the 120-second verifier
    timeout for storage-scale padding. 64 MiB of leading whitespace is fast
    (<1 s of parsing) only when the prefix is discarded, and stays well
    inside the 30-second child harness timeout.
    """
    app, logs, s = case(tmp_path)
    e = app / "evidence/answer.txt"
    answer = (
        " \n" * (32 * 1024 * 1024)
        + json.dumps(
            {
                "schema_version": "1",
                "task_id": s["task_id"],
                "result": s["result"],
                "limitations": s["limitations"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    e.write_text(answer)
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")
    started = time.monotonic()
    r = run(app, logs)
    elapsed = time.monotonic() - started
    assert r["evidence"] == 1.0 and r["aggregate_reward"] == 1.0
    # Retaining the prefix makes parsing quadratic: 64 MiB of padding takes
    # ~15 s in-process before the fix, vs well under 1 s after. 15 seconds
    # cleanly separates streaming discard from unbounded buffer accumulation
    # with generous headroom for slow runners.
    assert elapsed < 15.0


def test_evidence_leading_garbage_still_rejected(tmp_path):
    """Non-whitespace before the JSON value fails evidence like json.load."""
    app, logs, s = case(tmp_path)
    e = app / "evidence/answer.txt"
    e.write_text(
        " \n" * 1024
        + "garbage"
        + json.dumps(
            {
                "schema_version": "1",
                "task_id": s["task_id"],
                "result": s["result"],
                "limitations": s["limitations"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")
    r = run(app, logs)
    assert r["evidence"] == 0.0 and r["aggregate_reward"] == 0.0


def test_evidence_trailing_garbage_still_rejected(tmp_path):
    """Non-whitespace after the JSON value fails evidence like json.load."""
    app, logs, s = case(tmp_path)
    e = app / "evidence/answer.txt"
    e.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "task_id": s["task_id"],
                "result": s["result"],
                "limitations": s["limitations"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        + "garbage"
    )
    s["evidence"][0]["sha256"] = "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest()
    (app / "submission.json").write_text(json.dumps(s) + "\n")
    r = run(app, logs)
    assert r["evidence"] == 0.0 and r["aggregate_reward"] == 0.0
