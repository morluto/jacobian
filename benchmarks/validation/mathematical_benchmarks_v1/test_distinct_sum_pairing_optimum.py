from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from benchmarks.tooling.public_contract import check
from benchmarks.validation._source_module import load_source_module
from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures
from jsonschema import Draft202012Validator

TASK = "distinct-sum-pairing-optimum"
TASK_ROOT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/datasets/mathematical-benchmarks-v1"
    / TASK
)
PUBLIC_INPUT = json.loads((TASK_ROOT / "environment/input.json").read_text())
GROUND = PUBLIC_INPUT["ground_set"]


def test_result_only_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(tmp_path, TASK)


def _public_optimum(reverse: bool) -> list[list[int]]:
    # Vertex deletion plus sum masks is independent of the verifier's edge-subset
    # search. Both tie orders solve the entire public finite optimization problem.
    @cache
    def solve(remaining: int, sums: int) -> tuple[tuple[int, int], ...]:
        if not remaining:
            return ()
        first = remaining & -remaining
        a = first.bit_length()
        tail = remaining ^ first
        best = solve(tail, sums)
        for b in sorted(GROUND, reverse=reverse):
            bit = 1 << (b - 1)
            total = a + b
            if b <= a or not tail & bit or total > PUBLIC_INPUT["n"]:
                continue
            if sums & (1 << total):
                continue
            candidate = ((a, b), *solve(tail ^ bit, sums | (1 << total)))
            if len(candidate) > len(best):
                best = candidate
        return best

    return [list(pair) for pair in solve((1 << len(GROUND)) - 1, 0)]


@pytest.fixture
def support(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load_source_module(
        "_pairing_bounds_support", TASK_ROOT / "tests/verifier_support.py"
    )
    monkeypatch.setattr(
        module._load_public_contract,
        "__defaults__",
        (TASK_ROOT / "tests/public_contract.json",),
    )
    return module


def _schema_valid(submission: dict[str, Any]) -> bool:
    schema = json.loads((TASK_ROOT / "environment/submission_schema.json").read_text())
    return Draft202012Validator(schema).is_valid(submission)


def _assert_replay(
    tmp_path: Path,
    support: ModuleType,
    pairs: list[Any],
    *,
    schema_valid: bool,
    reward: float,
) -> None:
    submission = {"result": {"pairs": pairs}}
    assert _schema_valid(submission) is schema_valid
    assert support.submission_matches_public_schema(submission) is schema_valid
    app, logs = tmp_path / "app", tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    (app / "input.json").write_bytes(
        (TASK_ROOT / "environment/input.json").read_bytes()
    )
    path = app / "submission.json"
    path.write_text(json.dumps(submission))
    assert support.load_submission(path, require_input_binding=False) == (
        submission if schema_valid else None
    )
    replay = run_verifier_in_child(task=TASK_ROOT, app=app, logs=logs)
    assert replay.reward == reward
    assert replay.details["correctness"] == reward
    assert json.loads((logs / "reward.json").read_text()) == {"reward": reward}


def test_public_domain_and_generated_contract(support: ModuleType) -> None:
    assert list(range(1, 16)) == GROUND
    assert PUBLIC_INPUT["n"] == 15
    assert len(GROUND) // 2 == 7
    for a in GROUND:
        for b in GROUND:
            submission = {"result": {"pairs": [[a, b]]}}
            assert _schema_valid(submission)
            assert support.submission_matches_public_schema(submission)
    assert check(TASK_ROOT / "tests/public_contract.json", TASK_ROOT) == []


@pytest.mark.parametrize("reverse", [False, True])
def test_independent_alternate_optima_are_accepted(
    tmp_path: Path, support: ModuleType, reverse: bool
) -> None:
    pairs = _public_optimum(reverse)
    other = _public_optimum(not reverse)
    assert pairs != other
    assert len(pairs) == len(other)
    _assert_replay(tmp_path, support, pairs, schema_valid=True, reward=1.0)


@pytest.mark.parametrize(
    "mutation",
    [
        "seven-pairs",
        "empty",
        "missing",
        "reverse",
        "swap-members",
        "duplicate",
        "same-member",
        "sum-overflow",
    ],
)
def test_bounded_false_mathematics_still_reaches_replay(
    tmp_path: Path, support: ModuleType, mutation: str
) -> None:
    pairs = _public_optimum(False)
    if mutation == "seven-pairs":
        pairs = [[a, a + 1] for a in range(1, 15, 2)]
    elif mutation == "empty":
        pairs = []
    elif mutation == "missing":
        pairs.pop()
    elif mutation == "reverse":
        pairs.reverse()
    elif mutation == "swap-members":
        pairs[0].reverse()
    elif mutation == "duplicate":
        pairs.append(pairs[0])
    elif mutation == "same-member":
        pairs = [[1, 1]]
    else:
        pairs = [[14, 15]]
    _assert_replay(tmp_path, support, pairs, schema_valid=True, reward=0.0)


@pytest.mark.parametrize(
    "pairs",
    [
        [[1, 2]] * 8,
        [[1, 2]] * 1000,
        [[0, 2]],
        [[1, 16]],
        [[-(10**1000), 2]],
        [[1, 10**1000]],
        [[True, 2]],
        [[1, False]],
        [[1.5, 2]],
        [[1, "2"]],
        [[1]],
        [[1, 2, 3]],
    ],
)
def test_public_packing_scalar_and_shape_limits(
    tmp_path: Path, support: ModuleType, pairs: list[Any]
) -> None:
    _assert_replay(tmp_path, support, pairs, schema_valid=False, reward=0.0)
