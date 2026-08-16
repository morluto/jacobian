from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "permutation-inversion-involution"


def _case(tmp_path: Path):
    return _fixtures._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    _fixtures._write_json(app / "submission.json", {"result": submission["result"]})


def test_oracle_passes(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 1.0


def test_accepts_alternative_trace_set(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"] = list(reversed(submission["result"]["traces"]))
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_locally_plausible_wrong_transform(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["transformation"] = "REVERSE_POSITIONS"
    submission["result"]["value_multiplier"] = 1
    submission["result"]["value_offset"] = 0
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_accepts_reverse_position_involution(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = submission["result"]
    result["transformation"] = "REVERSE_POSITIONS"
    result["value_multiplier"] = 1
    result["value_offset"] = 0
    for trace in result["traces"]:
        permutation = trace["permutation"]
        transformed = list(reversed(permutation))
        trace["transformed"] = transformed
        trace["transformed_inversions"] = sum(
            transformed[i] > transformed[j] for i in range(7) for j in range(i + 1, 7)
        )
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_rejects_corrupted_trace(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][3]["inversions"] += 1
    _rewrite(app, submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0


def test_unhashable_permutation_trace_emits_zero_reward(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][0]["permutation"] = [1, 2, 3, 4, 5, 6, [7]]
    _rewrite(app, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_boolean_permutation_entry_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][0]["permutation"] = [True, 2, 3, 4, 5, 6, 7]
    _rewrite(app, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_boolean_result_field_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["fixed_point_count"] = False
    _rewrite(app, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_boolean_trace_inversions_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][0]["inversions"] = False
    _rewrite(app, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_boolean_trace_transformed_entry_is_rejected(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["traces"][0]["transformed"][0] = True
    _rewrite(app, submission)
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0
