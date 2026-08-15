import json
from itertools import combinations
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)


def _is_topology(value: object, n: int) -> bool:
    if (
        not isinstance(value, list)
        or any(type(item) is not int for item in value)
        or value != sorted(set(value))
    ):
        return False
    opens = set(value)
    full = (1 << n) - 1
    if (
        0 not in opens
        or full not in opens
        or any(item < 0 or item > full for item in opens)
    ):
        return False
    return all((a | b) in opens and (a & b) in opens for a in opens for b in opens)


def _all_topologies(n: int):
    middle = list(range(1, (1 << n) - 1))
    for size in range(len(middle) + 1):
        for chosen in combinations(middle, size):
            family = [0, *chosen, (1 << n) - 1]
            if _is_topology(family, n):
                yield set(family)


def _generated(inputs: list[list[int]], n: int) -> list[int] | None:
    required = set().union(*(set(item) for item in inputs))
    containing = [topology for topology in _all_topologies(n) if required <= topology]
    if not containing:
        return None
    return sorted(set.intersection(*containing))


def _result_valid(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "universe_size",
        "input_topologies",
        "generated_topology",
        "common_subtopology",
        "witness_open_set",
        "diagnosis",
    }:
        return False
    n = result["universe_size"]
    inputs = result["input_topologies"]
    generated = result["generated_topology"]
    common = result["common_subtopology"]
    witness = result["witness_open_set"]
    if type(n) is not int or n not in {3, 4} or not isinstance(inputs, list):
        return False
    if len(inputs) < 2 or any(
        not _is_topology(item, n) or len(item) < 4 for item in inputs
    ):
        return False
    families = [set(item) for item in inputs]
    if len({tuple(item) for item in inputs}) != len(inputs):
        return False
    if not all(
        not (left <= right) and not (right <= left)
        for left, right in combinations(families, 2)
    ):
        return False
    expected_generated = _generated(inputs, n)
    if generated != expected_generated or not _is_topology(common, n):
        return False
    common_set = set(common)
    generated_set = set(generated)
    return bool(
        all(common_set <= family for family in families)
        and common_set != generated_set
        and type(witness) is int
        and witness in generated_set
        and witness not in common_set
        and result["diagnosis"]
        == "COMMON_SUBTOPOLOGY_DOES_NOT_ESTABLISH_LEAST_CONTAINING_TOPOLOGY"
    )


def main() -> None:
    submission = load_submission()
    result = submission.get("result") if isinstance(submission, dict) else None
    mathematical = _result_valid(result)
    correct = bool(mathematical)
    reward = float(correct)
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(mathematical),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
