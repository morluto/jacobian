from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT / "benchmarks/datasets/conjecture-probes-v1/vizing-bounded-cartesian-products"
)


def _case(tmp_path: Path):
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    (app / "evidence").mkdir()
    evidence = TASK / "solution/evidence/answer.txt"
    shutil.copy2(evidence, app / "evidence/answer.txt")
    submission = json.loads((TASK / "solution/submission.json").read_text())
    return app, logs, submission


def _write(app: Path, submission: dict) -> None:
    evidence = app / "evidence/answer.txt"
    payload = {
        "schema_version": "1",
        "task_id": "jacobian/vizing-bounded-cartesian-products",
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


def test_oracle_certificate_gets_full_reward(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    _write(app, submission)
    reward = _run(app, logs)
    assert reward.details["aggregate_reward"] == 1.0
    assert reward.details["mathematics"] == 1.0


def test_alternate_minimum_witnesses_pass(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    for row in submission["result"]["graphs"]:
        row["minimum_dominating_set"] = list(reversed(row["minimum_dominating_set"]))
    for row in submission["result"]["pairs"]:
        row["left_minimum_dominating_set"] = list(
            reversed(row["left_minimum_dominating_set"])
        )
        row["right_minimum_dominating_set"] = list(
            reversed(row["right_minimum_dominating_set"])
        )
        row["product_minimum_dominating_set"] = list(
            reversed(row["product_minimum_dominating_set"])
        )
    _write(app, submission)
    assert _run(app, logs).details["aggregate_reward"] == 1.0


def test_wrong_values_omissions_duplicates_and_boolean_integers_fail(
    tmp_path: Path,
) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["graphs"][0]["domination_number"] += 1
    _write(app, submission)
    assert _run(app, logs).details["mathematics"] == 0.0
    app, logs, submission = _case(tmp_path / "missing")
    submission["result"]["pairs"].pop()
    _write(app, submission)
    assert _run(app, logs).details["mathematics"] == 0.0
    app, logs, submission = _case(tmp_path / "bool")
    submission["result"]["graphs"][0]["vertex_count"] = True
    _write(app, submission)
    assert _run(app, logs).details["mathematics"] == 0.0


def test_tampered_input_and_bad_evidence_binding_are_separate(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    input_data = json.loads((app / "input.json").read_text())
    input_data["graphs"][0]["adjacency"][0].append(1)
    (app / "input.json").write_text(json.dumps(input_data))
    _write(app, submission)
    reward = _run(app, logs)
    assert (
        reward.details["input_binding"] == 0.0
        and reward.details["aggregate_reward"] == 0.0
    )
    app, logs, submission = _case(tmp_path / "evidence")
    submission["witness"][0]["path"] = "evidence/../answer.txt"
    _write(app, submission)
    reward = _run(app, logs)
    assert (
        reward.details["witness_validity"] == 0.0
        and reward.details["aggregate_reward"] == 0.0
    )


def test_malformed_bound_evidence_fails_closed(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    _write(app, submission)
    evidence = app / "evidence/answer.txt"
    evidence.write_text("{")
    submission["witness"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    (app / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")
    reward = _run(app, logs)
    assert reward.details["witness_validity"] == 0.0
    assert reward.details["aggregate_reward"] == 0.0


def test_reward_emission_is_deterministic_for_malformed_json(tmp_path: Path) -> None:
    app, logs, _ = _case(tmp_path)
    (app / "submission.json").write_text("{")
    first = _run(app, logs)
    logs2 = tmp_path / "logs2"
    logs2.mkdir()
    second = _run(app, logs2)
    assert first == second
