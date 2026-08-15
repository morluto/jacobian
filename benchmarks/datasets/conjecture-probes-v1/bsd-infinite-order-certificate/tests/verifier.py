"""Verifier for a bounded Lutz-Nagell infinite-order certificate."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    is_regular_bounded_file,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    resolve_evidence,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/bsd-infinite-order-certificate"


def _rat(value: object) -> Fraction:
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError
    number = Fraction(value)
    if (
        str(number) != value
        or len(str(abs(number.numerator))) > 80
        or len(str(number.denominator)) > 80
    ):
        raise ValueError
    return number


def _point(value: object) -> tuple[Fraction, Fraction]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError
    return _rat(value[0]), _rat(value[1])


def _add(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], a: int
) -> tuple[Fraction, Fraction]:
    x1, y1 = left
    x2, y2 = right
    if left == right:
        if y1 == 0:
            raise ValueError
        slope = (3 * x1 * x1 + a) / (2 * y1)
    else:
        if x1 == x2:
            raise ValueError
        slope = (y2 - y1) / (x2 - x1)
    x3 = slope * slope - x1 - x2
    return x3, slope * (x1 - x3) - y1


def _mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "A",
        "B",
        "point",
        "discriminant",
        "y_square",
        "y_square_divides_discriminant",
        "double",
        "triple",
        "order_conclusion",
    }:
        return False
    a, b = result.get("A"), result.get("B")
    point = result.get("point")
    if (
        type(a) is not int
        or type(b) is not int
        or not (-20 <= a <= 20 and -20 <= b <= 20)
        or not isinstance(point, list)
        or len(point) != 2
        or any(type(v) is not int or not -100 <= v <= 100 for v in point)
    ):
        return False
    x, y = point
    discriminant = -16 * (4 * a**3 + 27 * b**2)
    if discriminant == 0 or y == 0 or y * y != x**3 + a * x + b:
        return False
    try:
        double = _add((Fraction(x), Fraction(y)), (Fraction(x), Fraction(y)), a)
        triple = _add(double, (Fraction(x), Fraction(y)), a)
        submitted_double, submitted_triple = (
            _point(result["double"]),
            _point(result["triple"]),
        )
    except (ValueError, ZeroDivisionError, TypeError):
        return False
    obstruction = abs(discriminant) % (y * y) != 0
    submitted_discriminant = result["discriminant"]
    submitted_y_square = result["y_square"]
    return (
        type(submitted_discriminant) is int
        and submitted_discriminant == discriminant
        and type(submitted_y_square) is int
        and submitted_y_square == y * y
        and result["y_square_divides_discriminant"] is False
        and obstruction
        and submitted_double == double
        and submitted_triple == triple
        and result["order_conclusion"] == "INFINITE_BY_LUTZ_NAGELL"
    )


def _reward(value: dict[str, Any]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def _raw_submission() -> dict[str, Any] | None:
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _witness_matches_result(witness: object, result: object) -> bool:
    if not isinstance(witness, list) or len(witness) != 1:
        return False
    target = resolve_evidence(witness[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    payload = read_evidence_json(witness[0], expected_path="evidence/answer.txt")
    return bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and json_value_equal(payload.get("result"), result)
    )


def main() -> None:
    input_bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol = isinstance(submission, dict)
    mathematics = bool(protocol and _mathematics(submission.get("result")))
    witness = bool(
        protocol
        and _witness_matches_result(submission.get("witness"), submission.get("result"))
    )
    aggregate = float(input_bound and protocol and mathematics and witness)
    _reward(
        {
            "protocol": float(protocol),
            "input_binding": 1.0 if input_bound else 0.0,
            "mathematics": 1.0 if mathematics else 0.0,
            "witness_validity": float(witness),
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
                "witness_validity": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
