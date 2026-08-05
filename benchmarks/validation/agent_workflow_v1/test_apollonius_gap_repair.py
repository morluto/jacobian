from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "apollonius-gap-repair"


def _load(app: Path) -> dict[str, object]:
    return json.loads((app / "submission.json").read_text())


def test_accepts_alternative_normalization(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    result = sub["result"]
    result.update(
        {
            "k": "1/2",
            "c": "4",
            "p": "4/3",
            "q": "-4",
            "center": "-4/3",
            "radius": "8/3",
            "circle_coefficients": ["1", "1", "8/3", "-16/3"],
            "distance_coefficients": ["3/4", "3/4", "2", "-4"],
            "multiplier": "3/4",
        }
    )
    support._write_json(app / "submission.json", sub)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_corrupt_proportionality(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    sub = _load(app)
    sub["result"]["distance_coefficients"][2] = "23"
    support._write_json(app / "submission.json", sub)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0
