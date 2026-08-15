from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

TESTS = Path("/tests")
LOGS = Path("/logs/verifier")

_PROJECTION_SUMMARY_KEYS = (
    "component_count",
    "nonempty_rows",
    "nonempty_columns",
    "witness_projection_cost",
    "universal_projection_cost",
    "universal_edge_bound",
)


def _write_reward(values: dict[str, object]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.json").write_text(json.dumps(values), encoding="utf-8")
    normalize_reward_file(LOGS / "reward.json")


def _source() -> dict | None:
    try:
        value = json.loads((TESTS / "input.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {
        "task_id",
        "source",
        "set_size",
        "allowed_ratios",
        "integer_bounds",
        "claimed_maximum",
    }:
        return None
    source = value.get("source")
    return value if isinstance(source, dict) and source.get("row") == 60 else None


def _factor(value: int) -> tuple[int, int, int]:
    core, two, three = value, 0, 0
    while core % 2 == 0:
        core //= 2
        two += 1
    while core % 3 == 0:
        core //= 3
        three += 1
    return core, two, three


def _minimum_projection_cost(total: int) -> int:
    cell_cost = [0] + [
        min(r + c for r in range(1, m + 1) for c in range(1, m + 1) if r * c >= m)
        for m in range(1, total + 1)
    ]
    dp = [0] + [2 * total + 1] * total
    for n in range(1, total + 1):
        dp[n] = min(dp[n - m] + cell_cost[m] for m in range(1, n + 1))
    return dp[total]


def _numbers_valid(numbers: object, source: dict) -> bool:
    bounds = source["integer_bounds"]
    return bool(
        isinstance(numbers, list)
        and len(numbers) == source["set_size"]
        and all(type(number) is int for number in numbers)
        and all(bounds["minimum"] <= number <= bounds["maximum"] for number in numbers)
        and len(set(numbers)) == len(numbers)
    )


def _pair_set(numbers: list[int], ratios: list[int]) -> set[tuple[int, int]]:
    return {
        (i, j)
        for i, left in enumerate(numbers)
        for j in range(i + 1, len(numbers))
        if max(left, numbers[j]) in {ratio * min(left, numbers[j]) for ratio in ratios}
    }


def _pairs_valid(submitted: object, actual: set[tuple[int, int]]) -> bool:
    if not isinstance(submitted, list):
        return False
    if any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(type(index) is not int for index in pair)
        or not 0 <= pair[0] < pair[1] < 100
        for pair in submitted
    ):
        return False
    pairs = [tuple(pair) for pair in submitted]
    return len(pairs) == 180 and len(set(pairs)) == 180 and set(pairs) == actual


def _factor_map_valid(
    submitted: object, numbers: list[int]
) -> dict[int, tuple[int, int, int]] | None:
    if not isinstance(submitted, list) or len(submitted) != len(numbers):
        return None
    factor_map: dict[int, tuple[int, int, int]] = {}
    required = {"index", "core", "two_exponent", "three_exponent"}
    for item in submitted:
        if not isinstance(item, dict) or set(item) != required:
            return None
        if any(type(item[key]) is not int for key in item):
            return None
        index = item["index"]
        declared = (item["core"], item["two_exponent"], item["three_exponent"])
        if index in factor_map or not 0 <= index < len(numbers):
            return None
        if declared != _factor(numbers[index]) or math.gcd(item["core"], 6) != 1:
            return None
        factor_map[index] = declared
    return factor_map if set(factor_map) == set(range(len(numbers))) else None


def _projection_summary(
    factor_map: dict[int, tuple[int, int, int]], total: int
) -> dict[str, int]:
    points: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for core, two, three in factor_map.values():
        points[core].add((two, three))
    rows = sum(len({three for _, three in component}) for component in points.values())
    columns = sum(len({two for two, _ in component}) for component in points.values())
    universal_cost = _minimum_projection_cost(total)
    return {
        "component_count": len(points),
        "nonempty_rows": rows,
        "nonempty_columns": columns,
        "witness_projection_cost": rows + columns,
        "universal_projection_cost": universal_cost,
        "universal_edge_bound": 2 * total - universal_cost,
    }


def _projection_summary_shape_valid(summary: object) -> bool:
    if not isinstance(summary, dict) or set(summary) != set(_PROJECTION_SUMMARY_KEYS):
        return False
    return (
        type(summary["component_count"]) is int
        and 1 <= summary["component_count"] <= 100
        and type(summary["nonempty_rows"]) is int
        and 1 <= summary["nonempty_rows"] <= 100
        and type(summary["nonempty_columns"]) is int
        and 1 <= summary["nonempty_columns"] <= 100
        and type(summary["witness_projection_cost"]) is int
        and 2 <= summary["witness_projection_cost"] <= 200
        and summary["universal_projection_cost"] == 20
        and type(summary["universal_projection_cost"]) is int
        and summary["universal_edge_bound"] == 180
        and type(summary["universal_edge_bound"]) is int
    )


def _result_shape_valid(result: object, source: dict) -> bool:
    required = {
        "numbers",
        "good_pairs",
        "factorizations",
        "projection_summary",
        "claimed_maximum",
        "conclusion",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    pairs = result["good_pairs"]
    return bool(
        _numbers_valid(result["numbers"], source)
        and isinstance(pairs, list)
        and len(pairs) == 180
        and all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(type(index) is int and 0 <= index <= 99 for index in pair)
            for pair in pairs
        )
        and len({tuple(pair) for pair in pairs}) == 180
        and isinstance(result["factorizations"], list)
        and len(result["factorizations"]) == source["set_size"]
        and all(
            isinstance(item, dict)
            and set(item) == {"index", "core", "two_exponent", "three_exponent"}
            and type(item["index"]) is int
            and 0 <= item["index"] <= 99
            and type(item["core"]) is int
            and 1 <= item["core"] <= 100000000
            and type(item["two_exponent"]) is int
            and 0 <= item["two_exponent"] <= 26
            and type(item["three_exponent"]) is int
            and 0 <= item["three_exponent"] <= 17
            for item in result["factorizations"]
        )
        and _projection_summary_shape_valid(result["projection_summary"])
        and type(result["claimed_maximum"]) is int
        and result["claimed_maximum"] == source["claimed_maximum"]
        and result["conclusion"] == "EXACT_MAXIMUM_CERTIFIED"
    )


def _result_valid(result: object, source: dict) -> bool:
    required = {
        "numbers",
        "good_pairs",
        "factorizations",
        "projection_summary",
        "claimed_maximum",
        "conclusion",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    if not _projection_summary_shape_valid(result["projection_summary"]):
        return False
    if type(result["claimed_maximum"]) is not int:
        return False
    numbers = result["numbers"]
    if not _numbers_valid(numbers, source):
        return False
    assert isinstance(numbers, list)

    actual_pairs = _pair_set(numbers, source["allowed_ratios"])
    if not _pairs_valid(result["good_pairs"], actual_pairs):
        return False
    factor_map = _factor_map_valid(result["factorizations"], numbers)
    if factor_map is None:
        return False
    return (
        result["projection_summary"] == _projection_summary(factor_map, len(numbers))
        and len(actual_pairs) == source["claimed_maximum"]
        and result["claimed_maximum"] == source["claimed_maximum"]
        and result["conclusion"] == "EXACT_MAXIMUM_CERTIFIED"
    )


def main() -> None:
    source = _source()
    submission = load_submission()
    protocol_ok = submission is not None
    correctness = bool(
        protocol_ok
        and source is not None
        and _result_valid(submission.get("result"), source)
    )
    reward = float(correctness)
    _write_reward(
        {
            "correctness": float(correctness),
            "reward": reward,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _write_reward(
            {
                "correctness": 0.0,
                "reward": 0.0,
            }
        )
