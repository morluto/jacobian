from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._solution import run_solution
from benchmarks.validation._verifier_child import VerifierOutput
from benchmarks.validation.mathematical_benchmarks_v1 import _verifier

TASK = (
    Path(__file__).parents[2]
    / "datasets"
    / "mathematical-benchmarks-v1"
    / "multiplicative-grid-extremum"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _case(tmp_path: Path) -> tuple[Path, Path, dict]:
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    run_solution(TASK, app, script="oracle.py", arguments=(str(app),))
    return app, logs, json.loads((app / "submission.json").read_text())


def _run(app: Path, logs: Path) -> VerifierOutput:
    return _verifier._run_verifier(TASK, app, logs)


def test_oracle_receives_full_reward(tmp_path: Path) -> None:
    app, logs, _ = _case(tmp_path)
    result = _run(app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 1.0


def test_alternative_coprime_rescaling_is_accepted(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["numbers"] = [
        5 * number for number in submission["result"]["numbers"]
    ]
    for factor in submission["result"]["factorizations"]:
        factor["core"] = 5
    _write_json(app / "submission.json", submission)
    assert _run(app, logs).reward == 1.0


def test_wrong_or_incomplete_edge_sets_are_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path / "float")
    submission["result"]["good_pairs"].pop()
    _write_json(app / "submission.json", submission)
    assert _run(app, logs).details["correctness"] == 0.0


def test_wrong_factorization_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["factorizations"][0]["core"] = 7
    _write_json(app / "submission.json", submission)
    assert _run(app, logs).details["correctness"] == 0.0


def test_wrong_projection_summary_is_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["projection_summary"]["nonempty_rows"] = 9
    _write_json(app / "submission.json", submission)
    assert _run(app, logs).details["correctness"] == 0.0


def test_boolean_and_float_attacks_are_rejected(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["result"]["numbers"][0] = True
    _write_json(app / "submission.json", submission)
    assert _run(app, logs).reward == 0.0

    app, logs, submission = _case(tmp_path / "float")
    submission["result"]["claimed_maximum"] = 180.0
    _write_json(app / "submission.json", submission)
    assert _run(app, logs).reward == 0.0


def test_extra_witness_and_input_tampering_fail_closed(tmp_path: Path) -> None:
    app, logs, submission = _case(tmp_path)
    submission["witness"] = [
        {"path": "evidence/answer.txt", "sha256": "sha256:" + "0" * 64}
    ]
    _write_json(app / "submission.json", submission)
    assert _run(app, logs).reward == 0.0

    app, logs, _ = _case(tmp_path / "input-tamper")
    source = json.loads((app / "input.json").read_text())
    source["claimed_maximum"] = 181
    _write_json(app / "input.json", source)
    assert _run(app, logs).reward == 0.0
