from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    witness_list_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
N = 12


def _popcount(state: int) -> int:
    return bin(state).count("1")


def _potential(state: int, weights: list[int], pair: int) -> int:
    heads = [index for index in range(N) if state & (1 << index)]
    return sum(weights[index] for index in heads) + pair * math.comb(len(heads), 2)


def _result(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "linear_weights",
        "pair_coefficient",
        "minimum_by_head_count",
        "nonterminal_delta",
        "state_count",
        "total_stopping_time",
        "average_stopping_time",
    }:
        return False
    weights = value["linear_weights"]
    pair = value["pair_coefficient"]
    if (
        not isinstance(weights, list)
        or len(weights) != N
        or any(type(item) is not int for item in weights)
        or type(pair) is not int
    ):
        return False
    potentials: list[int] = []
    layer_values = [[] for _ in range(N + 1)]
    for state in range(1 << N):
        current = _potential(state, weights, pair)
        potentials.append(current)
        heads = _popcount(state)
        layer_values[heads].append(current)
        if state == 0:
            if current != 0:
                return False
            continue
        successor = state ^ (1 << (heads - 1))
        if current <= 0 or _potential(successor, weights, pair) - current != -1:
            return False
    minima = [min(layer) for layer in layer_values]
    total = sum(potentials)
    average = Fraction(total, 1 << N)
    submitted_minima = value["minimum_by_head_count"]
    submitted_average = value["average_stopping_time"]
    if (
        not isinstance(submitted_minima, list)
        or any(type(item) is not int for item in submitted_minima)
        or type(value["nonterminal_delta"]) is not int
        or type(value["state_count"]) is not int
        or type(value["total_stopping_time"]) is not int
        or not isinstance(submitted_average, dict)
        or set(submitted_average) != {"numerator", "denominator"}
        or type(submitted_average["numerator"]) is not int
        or type(submitted_average["denominator"]) is not int
        or submitted_average["denominator"] <= 0
    ):
        return False
    return bool(
        submitted_minima == minima
        and minima == list(range(N + 1))
        and value["nonterminal_delta"] == -1
        and value["state_count"] == 1 << N
        and value["total_stopping_time"] == total
        and submitted_average["numerator"] * average.denominator
        == average.numerator * submitted_average["denominator"]
    )


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        data = json.loads(hidden)
        return bool(
            (WORKSPACE / "input.json").read_bytes() == hidden
            and data["source"]["revision"] == "882ba08befd0856f5364db1e53d58c7e2cf704f9"
            and data["source"]["row"] == 94
            and data["n"] == N
        )
    except (OSError, ValueError, KeyError):
        return False


def _evidence(value: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not witness_list_is_bound(value)
    ):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        return bool(path.read_text().strip())
    except (OSError, UnicodeError):
        return False


def _evaluate(submission: object) -> dict[str, float]:
    protocol_ok = isinstance(submission, dict)
    source_bound = _source_is_bound()
    result_correct = bool(protocol_ok and _result(submission.get("result")))
    evidence_valid = bool(protocol_ok and _evidence(submission.get("witness")))
    correct = bool(result_correct and source_bound)
    reward = aggregate_reward(
        correctness=correct,
        witness_validity=evidence_valid,
        protocol_ok=protocol_ok,
    )
    return {
        "correctness": float(correct),
        "witness_validity": float(evidence_valid),
        "reward": reward,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
