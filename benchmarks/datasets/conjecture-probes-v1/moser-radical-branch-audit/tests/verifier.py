from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

X = [
    (Fraction(1, 2), 0),
    (Fraction(-1, 2), 0),
    (Fraction(-1, 4), Fraction(-1, 12)),
    (0, 0),
    (Fraction(1, 4), Fraction(1, 12)),
    (Fraction(-1, 4), Fraction(1, 12)),
    (Fraction(1, 4), Fraction(-1, 12)),
]
CLAIMED = {
    (0, 1),
    (0, 4),
    (0, 6),
    (1, 2),
    (1, 5),
    (2, 3),
    (2, 5),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 6),
}


def _q(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError):
        return None


def _qadd(x, y):
    return (x[0] + y[0], x[1] + y[1])


def _qsquare(x):
    return (x[0] ** 2 + 33 * x[1] ** 2, 2 * x[0] * x[1])


def _vertical_square(left: str, right: str, corrupt: bool):
    left_sign = -1 if corrupt and left == "B5" else 1
    right_sign = -1 if corrupt and right == "B5" else 1
    left, right = (left.removesuffix("5"), right.removesuffix("5"))
    if left == right:
        if left_sign == right_sign:
            return (Fraction(0), Fraction(0))
        return {
            "A": (Fraction(17, 6), Fraction(1, 6)),
            "B": (Fraction(17, 6), Fraction(-1, 6)),
        }[left]
    key = tuple(sorted((left, right)))
    if "0" in key:
        return {
            ("0", "A"): (Fraction(17, 24), Fraction(1, 24)),
            ("0", "B"): (Fraction(17, 24), Fraction(-1, 24)),
            ("0", "T"): (Fraction(11, 4), Fraction(0)),
        }[key]
    differences = {
        ("A", "B"): (Fraction(1, 12), Fraction(0)),
        ("A", "T"): (Fraction(17, 24), Fraction(-1, 24)),
        ("B", "T"): (Fraction(17, 24), Fraction(1, 24)),
    }
    sums = {
        ("A", "B"): (Fraction(11, 4), Fraction(0)),
        ("A", "T"): (Fraction(149, 24), Fraction(1, 8)),
        ("B", "T"): (Fraction(149, 24), Fraction(-1, 8)),
    }
    return sums[key] if left_sign * right_sign < 0 else differences[key]


def _distances(corrupt: bool):
    tags = ["0", "0", "A", "T", "A", "B5", "B"]
    result = {}
    for i in range(7):
        for j in range(i + 1, 7):
            dx = (X[i][0] - X[j][0], X[i][1] - X[j][1])
            result[i, j] = _qadd(
                _qsquare(dx), _vertical_square(tags[i], tags[j], corrupt)
            )
    return result


def _table(
    value: Any, expected: dict[tuple[int, int], tuple[Fraction, Fraction]]
) -> bool:
    if not isinstance(value, list) or len(value) != 21:
        return False
    seen = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "pair",
            "distance_squared",
            "unit",
        }:
            return False
        pair, distance, unit = (row["pair"], row["distance_squared"], row["unit"])
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(x) is not int for x in pair)
            or (pair[0] == pair[1])
            or any(not 0 <= endpoint < 7 for endpoint in pair)
        ):
            return False
        if (
            not isinstance(distance, list)
            or len(distance) != 2
            or type(unit) is not bool
        ):
            return False
        parsed = tuple(_q(part) for part in distance)
        if any(part is None for part in parsed):
            return False
        key = tuple(sorted(pair))
        if key in seen or parsed != expected[key] or unit != (parsed == (1, 0)):
            return False
        seen.add(key)
    return seen == set(expected)


def mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "corrupted_pair_table",
        "false_claimed_edges",
        "repair",
        "corrected_pair_table",
        "corrected_edges",
    }:
        return False
    corrupt, fixed = (_distances(True), _distances(False))
    if not _table(result["corrupted_pair_table"], corrupt) or not _table(
        result["corrected_pair_table"], fixed
    ):
        return False
    false_edges = {pair for pair in CLAIMED if corrupt[pair] != (1, 0)}
    fixed_edges = {pair for pair, value in fixed.items() if value == (1, 0)}
    submitted_false = _edge_set(result["false_claimed_edges"], len(false_edges))
    submitted_fixed = _edge_set(result["corrected_edges"], len(fixed_edges))
    return (
        submitted_false == false_edges
        and result["repair"] == "FLIP_VERTEX_5_B_BRANCH_TO_POSITIVE"
        and (submitted_fixed == fixed_edges)
        and (len(fixed_edges) == 11)
    )


def _edge_set(value: Any, expected_count: int) -> set[tuple[int, int]] | None:
    if not isinstance(value, list) or len(value) != expected_count:
        return None
    normalized: set[tuple[int, int]] = set()
    for edge in value:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(endpoint) is not int for endpoint in edge)
            or (edge[0] == edge[1])
            or any(not 0 <= endpoint < 7 for endpoint in edge)
        ):
            return None
        normalized.add(tuple(sorted(edge)))
    return normalized if len(normalized) == expected_count else None


def _write(values):
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def main():
    input_bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol_ok = submission is not None
    math_ok = bool(protocol_ok and mathematics(submission.get("result")))
    reward = float(protocol_ok and input_bound and math_ok)
    _write(
        {
            "protocol_compliance": float(protocol_ok),
            "input_binding": float(input_bound),
            "correctness": float(math_ok),
            "reward": reward,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write(
            {
                "protocol_compliance": 0.0,
                "input_binding": 0.0,
                "correctness": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
