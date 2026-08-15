from __future__ import annotations

import json
import math
from fractions import Fraction
from functools import reduce
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
PLANES = {
    "cube_x0": (1, 0, 0, 0),
    "cube_x1": (1, 0, 0, 1),
    "cube_y0": (0, 1, 0, 0),
    "cube_y1": (0, 1, 0, 1),
    "cube_z0": (0, 0, 1, 0),
    "cube_z1": (0, 0, 1, 1),
    "tetra_A1C1D1": (0, 0, 1, 1),
    "tetra_BC1D1": (0, 1, -1, 0),
    "tetra_BA1D1": (1, 0, 1, 1),
    "tetra_BA1C1": (1, -1, 1, 1),
}
PLANE_LABELS = tuple(PLANES)
_EVIDENCE_STEMS = ("duplic", "restrict", "line", "region")


def _canonical(values: tuple[int, ...]) -> tuple[int, ...] | None:
    if not any(values[:-1]):
        return None
    divisor = reduce(math.gcd, (abs(v) for v in values if v), 0)
    normalized = tuple(v // divisor for v in values)
    first = next(v for v in normalized if v)
    return tuple(-v for v in normalized) if first < 0 else normalized


def _rational_line(
    values: tuple[Fraction, Fraction, Fraction],
) -> tuple[int, int, int] | None:
    a, b, c = values
    if a == b == 0:
        return None
    lcm = math.lcm(a.denominator, b.denominator, c.denominator)
    raw = (int(a * lcm), int(b * lcm), int(c * lcm))
    return _canonical(raw)


def _restriction(
    current: tuple[int, ...], prior: tuple[int, ...]
) -> tuple[int, int, int] | None:
    normal = current[:3]
    pivot = next(i for i, value in enumerate(normal) if value)
    free = [i for i in range(3) if i != pivot]
    p = Fraction(normal[pivot])
    q = prior[:3]
    coefficients = tuple(
        Fraction(q[index]) - Fraction(q[pivot] * normal[index], p) for index in free
    )
    rhs = Fraction(prior[3]) - Fraction(q[pivot] * current[3], p)
    return _rational_line((coefficients[0], coefficients[1], rhs))


def _line_regions(lines: set[tuple[int, int, int]]) -> int:
    regions = 1
    previous: list[tuple[int, int, int]] = []
    for a, b, c in sorted(lines):
        points: set[tuple[Fraction, Fraction]] = set()
        for d, e, f in previous:
            determinant = a * e - b * d
            if determinant:
                points.add(
                    (
                        Fraction(c * e - b * f, determinant),
                        Fraction(a * f - c * d, determinant),
                    )
                )
        regions += len(points) + 1
        previous.append((a, b, c))
    return regions


def _increments(order: list[tuple[str, tuple[int, ...]]]) -> list[int]:
    unique: list[tuple[int, ...]] = []
    increments: list[int] = []
    for _, plane in order:
        if plane in unique:
            increments.append(0)
            continue
        lines = {
            line for prior in unique if (line := _restriction(plane, prior)) is not None
        }
        increments.append(_line_regions(lines))
        unique.append(plane)
    return increments


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        source = json.loads(hidden)
        return bool(
            (WORKSPACE / "input.json").read_bytes() == hidden
            and source["source"]["revision"]
            == "c705198ae1043810b1e1693bd879250b51a7a523"
            and source["source"]["row"] == 20
        )
    except (OSError, ValueError, KeyError):
        return False


def _result(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "regions",
        "ordered_planes",
        "duplicate_groups",
    }:
        return False
    entries = value["ordered_planes"]
    if not isinstance(entries, list) or len(entries) != 10:
        return False
    order: list[tuple[str, tuple[int, ...]]] = []
    declared: list[int] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "label",
            "coefficients",
            "increment",
        }:
            return False
        label = entry["label"]
        coefficients = entry["coefficients"]
        increment = entry["increment"]
        if (
            not isinstance(label, str)
            or label not in PLANES
            or not isinstance(coefficients, list)
            or len(coefficients) != 4
            or any(type(item) is not int for item in coefficients)
            or type(increment) is not int
        ):
            return False
        canonical = _canonical(tuple(coefficients))
        if canonical != PLANES[label]:
            return False
        order.append((label, canonical))
        declared.append(increment)
    if {label for label, _ in order} != set(PLANES) or len(
        {label for label, _ in order}
    ) != 10:
        return False
    actual = _increments(order)
    duplicate = value["duplicate_groups"]
    valid_group = {"cube_z1", "tetra_A1C1D1"}
    return bool(
        type(value["regions"]) is int
        and declared == actual
        and value["regions"] == 1 + sum(actual) == 64
        and isinstance(duplicate, list)
        and len(duplicate) == 1
        and isinstance(duplicate[0], list)
        and len(duplicate[0]) == 2
        and all(type(member) is str for member in duplicate[0])
        and set(duplicate[0]) == valid_group
    )


def main() -> None:
    submission = load_submission()
    protocol_ok = submission is not None
    data = submission if isinstance(submission, dict) else {}
    math_ok = bool(protocol_ok and _source_is_bound() and _result(data.get("result")))
    reward = float(protocol_ok and math_ok)
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "reward": reward,
            },
            sort_keys=True,
        )
        + "\n"
    )
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
