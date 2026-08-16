from __future__ import annotations

import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/cycle-double-cover-multiplicity-audit"
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 4),
    (5, 7),
    (7, 9),
    (6, 9),
    (6, 8),
    (5, 8),
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
]
EDGE_INDEX = {tuple(sorted(edge)): i for i, edge in enumerate(EDGES)}


def _canonical(cycle: list[int]) -> tuple[int, ...]:
    variants = []
    for order in (cycle, list(reversed(cycle))):
        variants.extend(tuple(order[i:] + order[:i]) for i in range(len(order)))
    return min(variants)


def _multiplicities(value: object) -> list[int] | None:
    if not isinstance(value, list) or not 4 <= len(value) <= 12:
        return None
    seen: set[tuple[int, ...]] = set()
    counts = [0] * len(EDGES)
    for cycle in value:
        if (
            not isinstance(cycle, list)
            or not 5 <= len(cycle) <= 9
            or (not all(type(v) is int and 0 <= v < 10 for v in cycle))
            or (len(set(cycle)) != len(cycle))
        ):
            return None
        canonical = _canonical(cycle)
        if canonical in seen:
            return None
        seen.add(canonical)
        for i, left in enumerate(cycle):
            edge = tuple(sorted((left, cycle[(i + 1) % len(cycle)])))
            if edge not in EDGE_INDEX:
                return None
            counts[EDGE_INDEX[edge]] += 1
    return counts


def _exact_integer_list(value: object, expected: list[int]) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == len(expected)
        and all(type(item) is int for item in value)
        and (value == expected)
    )


def mathematics(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "flawed_cycles",
        "flawed_multiplicities",
        "non_double_edge_indices",
        "repair_cycles",
        "repair_multiplicities",
    }:
        return False
    flawed = _multiplicities(result["flawed_cycles"])
    repaired = _multiplicities(result["repair_cycles"])
    if flawed is None or repaired is None:
        return False
    bad = [i for i, count in enumerate(flawed) if count != 2]
    return (
        all(count >= 1 for count in flawed)
        and bool(bad)
        and _exact_integer_list(result["flawed_multiplicities"], flawed)
        and _exact_integer_list(result["non_double_edge_indices"], bad)
        and _exact_integer_list(result["repair_multiplicities"], repaired)
        and (repaired == [2] * len(EDGES))
    )


def _write(values: dict[str, object]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    reward_path = path / "reward.json"
    reward_path.write_text(json.dumps(values, sort_keys=True))
    normalize_reward_file(reward_path)


def main() -> None:
    submission = load_submission(require_input_binding=False)
    protocol = isinstance(submission, dict)
    mathematics_ok = bool(protocol and mathematics(submission.get("result")))
    input_bound = workspace_input_is_bound()
    reward = float(input_bound and protocol and mathematics_ok)
    _write(
        {
            "protocol": float(protocol),
            "input_binding": float(input_bound),
            "mathematics": float(mathematics_ok),
            "correctness": float(mathematics_ok),
            "aggregate_reward": reward,
            "reward": reward,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "correctness": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
