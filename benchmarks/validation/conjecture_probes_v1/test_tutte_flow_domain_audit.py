from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/tutte-flow-domain-audit"
TASK_ID = "jacobian/tutte-flow-domain-audit"
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 4),
    (5, 7),
    (7, 9),
    (6, 9),
    (6, 8),
    (5, 8),
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
]
FLAWED = [3, 2, 1, 4, 0, 1, 2, 4, 2, 1, 2, 1, 1, 2, 4]
REPAIR = [2, 1, 4, 3, 1, 1, 3, 3, 3, 1, 2, 1, 2, 1, 4]


def _balances(flow: list[int]) -> list[int]:
    values = [0] * 10
    for value, (source, target) in zip(flow, EDGES, strict=True):
        values[source] = (values[source] + value) % 5
        values[target] = (values[target] - value) % 5
    return values


def _result() -> dict[str, object]:
    return {
        "flawed_flow": FLAWED.copy(),
        "flawed_balances": _balances(FLAWED),
        "zero_edge_index": 4,
        "repair_flow": REPAIR.copy(),
        "repair_balances": _balances(REPAIR),
    }


def case(tmp_path: Path):
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment/input.json", app / "input.json")
    result = _result()
    submission = {"result": result}
    (app / "submission.json").write_text(json.dumps(submission) + "\n")
    return app, logs, submission


def write(app: Path, submission: dict, *, payload: object | None = None) -> None:
    del payload
    (app / "submission.json").write_text(
        json.dumps({"result": submission["result"]}) + "\n"
    )


def run(app: Path, logs: Path):
    return run_verifier_in_child(task=TASK, app=app, logs=logs)


def test_oracle_passes(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    assert run(app, logs).reward == 1.0


def test_rejects_zero_repair_and_noninteger_balance(tmp_path: Path) -> None:
    app, logs, submission = case(tmp_path)
    submission["result"]["repair_flow"] = FLAWED.copy()
    submission["result"]["repair_balances"] = [0] * 10
    write(app, submission)
    assert run(app, logs).reward == 0.0

    app, logs, submission = case(tmp_path / "type")
    submission["result"]["flawed_balances"][0] = False
    write(app, submission)
    assert run(app, logs).reward == 0.0


def test_result_only_submission_and_input_binding_are_hard_gates(
    tmp_path: Path,
) -> None:
    app, logs, submission = case(tmp_path)
    assert "witness" not in submission
    assert run(app, logs).reward == 1.0

    app, logs, _ = case(tmp_path / "input")
    (app / "input.json").write_text("{}\n")
    result = run(app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.details["mathematics"] == 1.0
    assert result.reward == 0.0


def test_malformed_submission_fails_closed(tmp_path: Path) -> None:
    app, logs, _ = case(tmp_path)
    (app / "submission.json").write_text('{"result":{},"result":{}}\n')
    assert run(app, logs).reward == 0.0
