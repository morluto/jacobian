from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "emerald-path-family-audit"


def _load(app: Path) -> dict[str, object]:
    return json.loads((app / "submission.json").read_text())


def test_accepts_alternative_family_member(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    result = submission["result"]
    assert isinstance(result, dict)
    result.update({"alpha": "5/4", "beta": "3/4", "odd_offset": "1/4"})
    for item in result["trace"]:
        x, y = item["x"], item["y"]
        from fractions import Fraction

        value = x * Fraction(5, 4) + y * Fraction(3, 4)
        item.update(
            {"value": str(value), "floor": value.numerator // value.denominator}
        )
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["reward"] == 1.0


def test_rejects_singleton_pair(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"].update({"alpha": "1", "beta": "1", "odd_offset": "0"})
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0


def test_rejects_corrupt_trace(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = _load(app)
    submission["result"]["trace"][9]["floor"] = 8
    support._write_json(app / "submission.json", submission)
    assert support._run_verifier(task, app, logs)["correctness"] == 0.0
