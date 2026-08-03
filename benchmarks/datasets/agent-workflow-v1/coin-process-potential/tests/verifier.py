from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/coin-process-potential"
CONCLUSION = "TERMINATION_AND_MEAN_CERTIFIED"
SCOPE = "all 4096 states of the frozen 12-coin process"
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
    submitted_average = value["average_stopping_time"]
    return bool(
        value["minimum_by_head_count"] == minima
        and minima == list(range(N + 1))
        and value["nonterminal_delta"] == -1
        and value["state_count"] == 1 << N
        and value["total_stopping_time"] == total
        and isinstance(submitted_average, dict)
        and set(submitted_average) == {"numerator", "denominator"}
        and submitted_average["numerator"] == average.numerator
        and submitted_average["denominator"] == average.denominator
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
        or not evidence_list_is_bound(value)
    ):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        text = path.read_text().casefold()
    except (OSError, UnicodeError):
        return False
    return all(word in text for word in ("potential", "decreases", "minimum", "mean"))


def _evaluate(submission: object) -> dict[str, float | bool]:
    protocol = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion=CONCLUSION,
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        protocol
        and isinstance(submission, dict)
        and _source_is_bound()
        and submission.get("scope") == SCOPE
        and submission.get("completeness") == "COMPLETE"
        and submission.get("claimed_assurance") == "COMPUTED"
        and _result(submission.get("result"))
        and _evidence(submission.get("evidence"))
        and any(
            "arbitrary n" in item.casefold()
            for item in submission.get("limitations", [])
        )
    )
    score = 1.0 if correct and not false_certification else 0.0
    return {
        "correctness": score,
        "evidence_validity": score,
        "scope_accuracy": score,
        "assurance_calibration": score,
        "reward": score,
        "false_certification": false_certification,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
