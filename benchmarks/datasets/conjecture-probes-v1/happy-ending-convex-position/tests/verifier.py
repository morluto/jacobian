"""Clean-room verifier for the finite Happy Ending convex-position probe."""

from __future__ import annotations

import itertools
import json
from collections import deque
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/happy-ending-convex-position"
SCOPE = "happy-ending-convex-position:points-v1"
RESULT_KEYS = frozenset(
    {
        "general_position",
        "convex_subset_counts",
        "maximum_convex_size",
        "maximum_witness_cyclic",
        "scope_identity",
    }
)


def _cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(set(points))
    lower: list[tuple[int, int]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _frozen() -> dict[str, Any] | None:
    try:
        value = json.loads(Path("/tests/input.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("task_id") != TASK_ID:
        return None
    points = value.get("points")
    return value if isinstance(points, list) and len(points) == 13 else None


def _cyclic(witness: list[tuple[int, int]]) -> bool:
    """Require the witness order to match the reconstructed convex hull.

    A same-sign local-turn test does not guarantee cyclic hull order because
    self-intersecting permutations can satisfy it. Compare the witness order
    with the independently reconstructed convex hull up to rotation and
    reversal instead.
    """
    if len(witness) < 3:
        return False
    hull = _hull(witness)
    if len(hull) != len(witness) or set(hull) != set(witness):
        return False
    targets = (hull, list(reversed(hull)))
    for target in targets:
        rotated = deque(witness)
        for _ in range(len(witness)):
            if list(rotated) == target:
                return True
            rotated.rotate(1)
    return False


def _convex_profile(
    records: list[dict[str, Any]],
) -> (
    tuple[list[str], list[tuple[int, int]], dict[int, int], int, set[frozenset[str]]]
    | None
):
    ids = [record["id"] for record in records]
    points = [(record["x"], record["y"]) for record in records]
    if not all(_cross(*triple) != 0 for triple in itertools.combinations(points, 3)):
        return None
    counts: dict[int, int] = {}
    maximum = 2
    maximum_sets: set[frozenset[str]] = set()
    for size in range(3, 14):
        count = 0
        for subset in itertools.combinations(range(13), size):
            selected = [points[index] for index in subset]
            if len(_hull(selected)) == size:
                count += 1
                if size > maximum:
                    maximum = size
                    maximum_sets.clear()
                if size == maximum:
                    maximum_sets.add(frozenset(ids[index] for index in subset))
        counts[size] = count
    return (ids, points, counts, maximum, maximum_sets)


def _submitted_counts(rows: Any) -> dict[int, int] | None:
    if not isinstance(rows, list) or len(rows) != 11:
        return None
    submitted: dict[int, int] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"size", "count"}
            or type(row["size"]) is not int
            or (type(row["count"]) is not int)
            or (row["size"] in submitted)
        ):
            return None
        submitted[row["size"]] = row["count"]
    return submitted


def _witness_coordinates(
    witness: Any, *, ids: list[str], points: list[tuple[int, int]], maximum: int
) -> list[tuple[int, int]] | None:
    if (
        not isinstance(witness, list)
        or len(witness) != maximum
        or len(set(witness)) != maximum
        or any(name not in ids for name in witness)
    ):
        return None
    return [points[ids.index(name)] for name in witness]


def _mathematics(result: Any, frozen: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or set(result) != RESULT_KEYS:
        return False
    profile = _convex_profile(frozen["points"])
    if profile is None or result.get("general_position") is not True:
        return False
    ids, points, counts, maximum, maximum_sets = profile
    submitted = _submitted_counts(result.get("convex_subset_counts"))
    coordinates = _witness_coordinates(
        result.get("maximum_witness_cyclic"), ids=ids, points=points, maximum=maximum
    )
    if submitted is None or coordinates is None:
        return False
    witness = result["maximum_witness_cyclic"]
    return (
        submitted == counts
        and result.get("maximum_convex_size") == maximum
        and (frozenset(witness) in maximum_sets)
        and _cyclic(coordinates)
        and (result.get("scope_identity") == SCOPE)
    )


def _reward(value: dict[str, Any]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def main() -> None:
    input_bound = workspace_input_is_bound()
    frozen = _frozen()
    submission = load_submission(require_input_binding=False)
    protocol = isinstance(submission, dict)
    mathematics = bool(
        frozen and protocol and _mathematics(submission.get("result"), frozen)
    )
    aggregate = float(input_bound and protocol and mathematics)
    _reward(
        {
            "protocol": float(protocol),
            "input_binding": float(input_bound),
            "mathematics": float(mathematics),
            "aggregate_reward": aggregate,
            "reward": aggregate,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _reward(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
