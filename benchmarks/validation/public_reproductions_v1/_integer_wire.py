"""Equivalent exact integer tokens reach the real Smith verifier unchanged."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.public_reproductions_v1._fixtures import _prepare_case


def check_equivalent_integer_tokens(
    tmp_path: Path, task_name: str, suffix: str
) -> None:
    task, app, logs = _prepare_case(tmp_path, task_name, "exact-numbers")
    path = app / "submission.json"
    result = json.loads(path.read_text())["result"]
    factors = ",".join(str(value) + suffix for value in result["invariant_factors"])
    path.write_text(
        '{"result":{"rank":'
        + str(result["rank"])
        + suffix
        + ',"invariant_factors":['
        + factors
        + "]}}"
    )
    replay = run_verifier_in_child(task=task, app=app, logs=logs)
    assert replay.reward == 1.0
    assert replay.details["correctness"] == 1.0


def check_invalid_integer_claim(tmp_path: Path, rank: str) -> None:
    task, app, logs = _prepare_case(tmp_path, "smith-rectangular", "wrong-numbers")
    (app / "submission.json").write_text(
        '{"result":{"rank":' + rank + ',"invariant_factors":[2,6]}}'
    )
    replay = run_verifier_in_child(task=task, app=app, logs=logs)
    assert replay.reward == 0.0
    assert replay.details["correctness"] == 0.0
