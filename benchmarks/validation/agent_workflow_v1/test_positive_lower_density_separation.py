from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from benchmarks.validation.agent_workflow_v1 import support

TASK = "positive-lower-density-separation"


def _run(tmp_path: Path, mutate=None):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    if mutate:
        mutate(submission)
        support._bind_result_evidence(app, submission)
        support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def _set_base(submission, b):
    levels = []
    for m in range(8):
        high, low = b ** (2 * m + 1), b ** (2 * m + 2)
        count = (low - 1) // (b + 1)
        levels.append(
            {
                "level": m,
                "included_endpoint": high,
                "excluded_endpoint": low,
                "cumulative_count": count,
                "included_density": str(Fraction(count, high)),
                "excluded_density": str(Fraction(count, low)),
            }
        )
    submission["result"].update(
        {
            "base": b,
            "levels": levels,
            "lower_density": str(Fraction(1, b + 1)),
            "upper_density": str(Fraction(b, b + 1)),
        }
    )


def test_oracle_and_alternative_base_pass(tmp_path: Path) -> None:
    assert _run(tmp_path / "oracle")["reward"] == 1.0
    assert _run(tmp_path / "alternative", lambda s: _set_base(s, 7))["reward"] == 1.0


def test_rejects_corrupted_endpoint_count(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["levels"][4]["cumulative_count"] += 1

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_equal_subsequence_limits(tmp_path: Path) -> None:
    def mutate(s):
        s["result"]["upper_density"] = s["result"]["lower_density"]

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_out_of_scope_base(tmp_path: Path) -> None:
    assert _run(tmp_path, lambda s: _set_base(s, 10))["reward"] == 0.0


def test_rejects_false_verified_claim(tmp_path: Path) -> None:
    def mutate(s):
        s["claimed_assurance"] = "VERIFIED"

    result = _run(tmp_path, mutate)
    assert result["false_certification"] is True
    assert result["reward"] == 0.0
