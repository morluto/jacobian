"""Hidden expected.json must not rewrite public-reproduction predicates."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from benchmarks.validation.public_reproductions_v1._fixtures import (
    _prepare_case,
    _write_json,
)
from benchmarks.validation.public_reproductions_v1._verifier import _run_verifier

_MUTATIONS = (
    ("reduced-point", "expected_free_ranks", []),
    ("integral-circle", "expected_free_ranks", []),
    ("smith-rectangular", "expected_rank", 0),
    ("smith-rank-deficient", "expected_rank", 99),
    (
        "gaussian-sixth-moment",
        "expected_moment",
        {"real": {"num": "0", "den": "1"}, "imaginary": {"num": "0", "den": "1"}},
    ),
    ("recurrence-fibonacci", "expected_first", "combinatorics.compute.lucas"),
    ("lean-retrieval", "expected_candidate_tactic", {"command": "sorry"}),
    ("lean-transition", "expected_goal_count", 99),
)


@pytest.mark.parametrize(("task_name", "field", "value"), _MUTATIONS)
def test_hidden_expected_mutation_does_not_change_reward(
    tmp_path: Path, task_name: str, field: str, value: object
) -> None:
    task, app, logs = _prepare_case(tmp_path, task_name, "expected-mutation")
    copied_task = tmp_path / f"{task_name}-expected"
    shutil.copytree(task, copied_task)
    expected_path = copied_task / "tests" / "expected.json"
    expected = json.loads(expected_path.read_text())
    if field not in expected:
        pytest.skip(f"{task_name} expected.json has no {field}")
    expected[field] = value
    _write_json(expected_path, expected)
    assert _run_verifier(copied_task, app, logs).reward == pytest.approx(1.0)
