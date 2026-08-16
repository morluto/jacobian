"""Rigorous rational-interval verifier for a finite Littlewood search."""

from __future__ import annotations

import json
from fractions import Fraction
from math import isqrt
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/littlewood-certified-finite-search"
SCALE = 10**80


def _encode(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _rat(value: object) -> Fraction | None:
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


def bounds(d, n):
    root = isqrt(d * SCALE * SCALE)
    lo = Fraction(root, SCALE)
    hi = Fraction(root + 1, SCALE)
    floor = isqrt(d * n * n)
    if 4 * d * n * n < (2 * floor + 1) ** 2:
        return (floor, floor, n * lo - floor, n * hi - floor)
    return (floor, floor + 1, floor + 1 - n * hi, floor + 1 - n * lo)


def row(n):
    a = bounds(2, n)
    b = bounds(3, n)
    return {
        "n": n,
        "floors": [a[0], b[0]],
        "nearest": [a[1], b[1]],
        "lower": n * a[2] * b[2],
        "upper": n * a[3] * b[3],
    }


def expected():
    rows = []
    best = None
    for n in range(1, 2001):
        current = row(n)
        hi = current["upper"]
        if best is None or hi < best["lower"]:
            rows.append(current)
            best = current
    assert best is not None and all(
        best["upper"] < row(n)["lower"] for n in range(1, 2001) if n != best["n"]
    )
    return (rows, best)


def mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "records",
        "argmin_n",
        "minimum_lower",
        "minimum_upper",
        "comparison_status",
    }:
        return False
    rows, best = expected()
    records = result.get("records")
    if not isinstance(records, list) or len(records) != len(rows):
        return False
    parsed = []
    for item in records:
        if not isinstance(item, dict) or set(item) != {
            "n",
            "floors",
            "nearest",
            "lower",
            "upper",
        }:
            return False
        lower = _rat(item["lower"])
        upper = _rat(item["upper"])
        if lower is None or upper is None:
            return False
        parsed.append(
            {
                "n": item["n"],
                "floors": item["floors"],
                "nearest": item["nearest"],
                "lower": lower,
                "upper": upper,
            }
        )
    return (
        parsed == rows
        and result.get("argmin_n") == best["n"]
        and (_rat(result.get("minimum_lower")) == best["lower"])
        and (_rat(result.get("minimum_upper")) == best["upper"])
        and (result.get("comparison_status") == "STRICTLY_SEPARATED_INTERVALS")
    )


def reward(v):
    p = Path("/logs/verifier")
    p.mkdir(parents=True, exist_ok=True)
    (p / "reward.json").write_text(json.dumps(v, sort_keys=True))
    normalize_reward_file(p / "reward.json")


def main():
    input_bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol = isinstance(submission, dict)
    mathematics_ok = bool(protocol and mathematics(submission.get("result")))
    aggregate = float(input_bound and protocol and mathematics_ok)
    reward(
        {
            "protocol": float(protocol),
            "input_binding": float(input_bound),
            "mathematics": float(mathematics_ok),
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
