from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    aggregate_reward,
    json_value_equal,
    load_submission,
    read_evidence_json,
    witness_list_is_bound,
    workspace_input_is_bound,
)

Q_PATTERN = re.compile(r"^-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?$")
MAX_Q_CHARS = 128


def _q(x):
    if not isinstance(x, str) or len(x) > MAX_Q_CHARS or Q_PATTERN.fullmatch(x) is None:
        raise ValueError
    q = Fraction(x)
    if str(q) != x:
        raise ValueError
    return q


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_json(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("out-of-range JSON number")
    return parsed


def _point(x):
    if not isinstance(x, list) or len(x) != 2:
        raise ValueError
    return _q(x[0]), _q(x[1])


def _param(t):
    return Fraction(2) * (1 - t * t) / (1 + t * t), Fraction(2) * t / (1 + t * t)


def _residuals(point):
    x, y = point
    return x * x / Fraction(4) + y * y - 1, -3 * x * y


def mathematics(result):
    if not isinstance(result, dict) or set(result) != {
        "finite_parameters",
        "finite_points",
        "missing_projective_parameter",
        "missing_point",
        "footpoint_records",
    }:
        return False
    try:
        params = [_q(x) for x in result["finite_parameters"]]
        points = [_point(x) for x in result["finite_points"]]
        missing = _point(result["missing_point"])
        projective = [_q(x) for x in result["missing_projective_parameter"]]
    except (ValueError, ZeroDivisionError):
        return False
    if (
        params != [Fraction(-1), Fraction(0), Fraction(1)]
        or points != sorted(_param(t) for t in params)
        or projective != [Fraction(1), Fraction(0)]
        or missing != (Fraction(-2), Fraction(0))
    ):
        return False
    expected_points = sorted([missing, *points])
    records = result["footpoint_records"]
    if not isinstance(records, list) or len(records) != 4:
        return False
    for row, point in zip(records, expected_points, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "point",
            "ellipse_residual",
            "normal_residual",
        }:
            return False
        try:
            submitted = _point(row["point"])
            er = _q(row["ellipse_residual"])
            nr = _q(row["normal_residual"])
        except (ValueError, ZeroDivisionError):
            return False
        if submitted != point or (er, nr) != _residuals(point) or er or nr:
            return False
    return True


def _write(values):
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))


def main():
    input_bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol_ok = submission is not None
    math_ok = bool(protocol_ok and mathematics(submission.get("result")))
    evidence_ok = bool(
        protocol_ok and witness_list_is_bound(submission.get("witness"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            submission["witness"][0],
            expected_path="evidence/answer.json",
            max_bytes=None,
        )
        if evidence_ok
        else None
    )
    evidence_ok = bool(
        isinstance(payload, dict)
        and payload.get("schema_version") == "1"
        and json_value_equal(payload.get("result"), submission.get("result"))
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=evidence_ok,
        protocol_ok=protocol_ok and input_bound,
    )
    _write(
        {
            "protocol_compliance": float(protocol_ok),
            "input_binding": float(input_bound),
            "correctness": float(math_ok),
            "witness_validity": float(evidence_ok),
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
                "witness_validity": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
