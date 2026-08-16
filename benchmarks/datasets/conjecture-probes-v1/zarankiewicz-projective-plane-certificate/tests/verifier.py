from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/zarankiewicz-projective-plane-certificate"
SCOPE = "pg2-f3-zarankiewicz-k22-extremal-certificate-v1"
LIMITATIONS = [
    "ONE_FINITE_PG2_F3_INSTANCE",
    "EXACT_K22_FREE_PAIR_COUNT_REPLAY",
    "NO_GENERAL_ZARANKIEWICZ_CONCLUSION",
]


def _triples(value: Any) -> list[tuple[int, int, int]] | None:
    if not isinstance(value, list) or len(value) != 13:
        return None
    out = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            return None
        if any(type(x) is not int or not 0 <= x < 3 for x in row):
            return None
        triple = tuple(row)
        if triple == (0, 0, 0):
            return None
        first = next(x for x in triple if x)
        if first != 1:
            return None
        out.append(triple)
    return out if len(set(out)) == 13 else None


def _all_projective_classes() -> set[tuple[int, int, int]]:
    out = set()
    for value in itertools.product(range(3), repeat=3):
        if value == (0, 0, 0):
            continue
        first = next(x for x in value if x)
        inv = 1 if first == 1 else 2
        out.add(tuple(inv * x % 3 for x in value))
    return out


def _edges(value: Any) -> set[tuple[int, int]] | None:
    if not isinstance(value, list) or len(value) != 52:
        return None
    out = set()
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            return None
        if any(type(x) is not int or not 0 <= x < 13 for x in row):
            return None
        out.add((row[0], row[1]))
    return out if len(out) == 52 else None


def _pair_rows(value: Any, neighbors: list[set[int]]) -> bool:
    if not isinstance(value, list) or len(value) != 78:
        return False
    seen = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {"pair", "common_neighbors"}:
            return False
        pair = row["pair"]
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(x) is not int or not 0 <= x < 13 for x in pair)
            or (pair[0] >= pair[1])
            or (type(row["common_neighbors"]) is not int)
        ):
            return False
        key = tuple(pair)
        if key in seen or row["common_neighbors"] != len(
            neighbors[pair[0]] & neighbors[pair[1]]
        ):
            return False
        seen.add(key)
    return seen == set(itertools.combinations(range(13), 2))


def _result_components(result: Any):
    if not isinstance(result, dict):
        return None
    required = {
        "points",
        "lines",
        "edges",
        "left_degrees",
        "right_degrees",
        "left_pair_common_counts",
        "right_pair_common_counts",
        "edge_count",
        "pair_budget",
        "excluded_edge_count",
    }
    if set(result) != required:
        return None
    points = _triples(result["points"])
    lines = _triples(result["lines"])
    edges = _edges(result["edges"])
    if points is None or lines is None or edges is None:
        return None
    return (points, lines, edges)


def _incidence_neighbors(
    points: list[tuple[int, int, int]],
    lines: list[tuple[int, int, int]],
    edges: set[tuple[int, int]],
):
    classes = _all_projective_classes()
    if set(points) != classes or set(lines) != classes:
        return None
    expected = {
        (i, j)
        for i, p in enumerate(points)
        for j, line in enumerate(lines)
        if sum((a * b for a, b in zip(p, line, strict=True))) % 3 == 0
    }
    if edges != expected:
        return None
    left = [{j for i2, j in edges if i2 == i} for i in range(13)]
    right = [{i for i, j2 in edges if j2 == j} for j in range(13)]
    left_degrees = [len(x) for x in left]
    right_degrees = [len(x) for x in right]
    if left_degrees != [4] * 13 or right_degrees != [4] * 13:
        return None
    return (left, right, left_degrees, right_degrees)


def mathematics(result: Any) -> bool:
    components = _result_components(result)
    if components is None:
        return False
    points, lines, edges = components
    incidence = _incidence_neighbors(points, lines, edges)
    if incidence is None:
        return False
    left, right, left_degrees, right_degrees = incidence
    if (
        result["left_degrees"] != left_degrees
        or result["right_degrees"] != right_degrees
    ):
        return False
    if any(
        (len(left[a] & left[b]) > 1 for a, b in itertools.combinations(range(13), 2))
    ):
        return False
    if any(
        (len(right[a] & right[b]) > 1 for a, b in itertools.combinations(range(13), 2))
    ):
        return False
    if not _pair_rows(result["left_pair_common_counts"], left):
        return False
    if not _pair_rows(result["right_pair_common_counts"], right):
        return False
    return (
        type(result["edge_count"]) is int
        and result["edge_count"] == 52
        and (type(result["pair_budget"]) is int)
        and (result["pair_budget"] == math.comb(13, 2))
        and (type(result["excluded_edge_count"]) is int)
        and (result["excluded_edge_count"] == 53)
        and (12 * math.comb(4, 2) + math.comb(5, 2) > math.comb(13, 2))
    )


def _reject_constant(value: str) -> None:
    raise ValueError(value)


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(value)
    return parsed


def _raw_submission() -> dict[str, Any] | None:
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(
            path.read_text(), parse_constant=_reject_constant, parse_float=_finite_float
        )
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def reward(value: dict[str, float]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def main() -> None:
    input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    contract = bool(submission)
    raw = _raw_submission()
    math_ok = bool(isinstance(raw, dict) and mathematics(raw.get("result")))
    protocol_ok = bool(contract)
    aggregate = float(input_binding and protocol_ok and math_ok)
    reward(
        {
            "input_binding": float(input_binding),
            "protocol": float(protocol_ok),
            "mathematics": float(math_ok),
            "aggregate_reward": aggregate,
            "reward": aggregate,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        reward(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
