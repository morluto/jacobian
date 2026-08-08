from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/mahler-leading-coefficient-audit"
SCOPE = "mahler-leading-coefficient-audit-v1"
LIMITATIONS = [
    "ONE_DEGREE_EIGHT_POLYNOMIAL",
    "EXACT_FACTOR_FORMULA_AUDIT_ONLY",
    "LEHMER_PROBLEM_NOT_ASSESSED",
]
TARGET = [2, -11, 21, -22, 23, -22, 21, -11, 2]
RATIONAL = re.compile(r"^-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?$")


def _q(value):
    if not isinstance(value, str) or RATIONAL.fullmatch(value) is None:
        raise ValueError
    return Fraction(value)


def _pair(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError
    return _q(value[0]), _q(value[1])


def _mul(x, y):
    return x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def _poly_mul(left, right):
    out = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[i + j] += a * b
    return out


def _json_equal(left: object, right: object) -> bool:
    pending = [(left, right, 0)]
    visited = 0
    while pending:
        current_left, current_right, depth = pending.pop()
        visited += 1
        if visited > 100_000 or depth > 128:
            return False
        if type(current_left) is not type(current_right):
            return False
        if isinstance(current_left, dict):
            if set(current_left) != set(current_right):
                return False
            pending.extend(
                (current_left[key], current_right[key], depth + 1)
                for key in current_left
            )
        elif isinstance(current_left, list):
            if len(current_left) != len(current_right):
                return False
            pending.extend(
                (a, b, depth + 1)
                for a, b in zip(current_left, current_right, strict=True)
            )
        elif current_left != current_right:
            return False
    return True


def _integral_json_number(value: object) -> int | None:
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _evidence_payload_matches_submission(payload: object, raw: object) -> bool:
    return bool(
        isinstance(payload, dict)
        and isinstance(raw, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and _json_equal(payload.get("task_id"), raw.get("task_id"))
        and _json_equal(payload.get("result"), raw.get("result"))
        and _json_equal(payload.get("limitations"), raw.get("limitations"))
    )


def _normalized_factors(value: object) -> list[list[int]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    normalized: list[list[int]] = []
    for factor in value:
        if not isinstance(factor, list) or len(factor) != 3:
            return None
        normalized_factor: list[int] = []
        for coefficient in factor:
            integral = _integral_json_number(coefficient)
            if integral is None:
                return None
            normalized_factor.append(integral)
        normalized.append(normalized_factor)
    return normalized


def _factorization_is_valid(factors: list[list[int]]) -> bool:
    product = [1]
    for factor in factors:
        if factor[0] <= 0 or math.gcd(*map(abs, factor)) != 1:
            return False
        product = _poly_mul(product, factor)
    return product == TARGET


def mathematics(result):
    if not isinstance(result, dict) or set(result) != {
        "factors",
        "outside_contributions",
        "flawed_monic_result",
        "leading_coefficient",
        "corrected_mahler_measure",
    }:
        return False
    factors = result["factors"]
    normalized_factors = _normalized_factors(factors)
    if normalized_factors is None:
        return False
    if normalized_factors != sorted(normalized_factors):
        return False
    if not _factorization_is_valid(normalized_factors):
        return False
    expected = {
        (1, -3, 1): (Fraction(3, 2), Fraction(1, 2)),
        (1, -1, 1): (Fraction(1), Fraction(0)),
        (1, 1, 1): (Fraction(1), Fraction(0)),
        (2, -5, 2): (Fraction(2), Fraction(0)),
    }
    contributions_raw = result["outside_contributions"]
    if not isinstance(contributions_raw, list) or len(contributions_raw) != 4:
        return False
    try:
        contributions = [_pair(v) for v in contributions_raw]
        flawed = _pair(result["flawed_monic_result"])
        corrected = _pair(result["corrected_mahler_measure"])
        leading = _q(result["leading_coefficient"])
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    if (
        contributions != [expected[tuple(f)] for f in normalized_factors]
        or leading != 2
    ):
        return False
    aggregate = (Fraction(1), Fraction(0))
    for contribution in contributions:
        aggregate = _mul(aggregate, contribution)
    return (
        flawed == aggregate
        and corrected == (leading * aggregate[0], leading * aggregate[1])
        and corrected == (Fraction(6), Fraction(2))
    )


def _raw():
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _write(values):
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(values, sort_keys=True))


def main():
    raw = _raw()
    submission = load_submission(require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="NONMONIC_FORMULA_REPAIRED",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    evidence_ok = bool(
        isinstance(raw, dict)
        and evidence_list_is_bound(raw.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            raw["evidence"][0], expected_path="evidence/answer.json", max_bytes=None
        )
        if evidence_ok
        else None
    )
    evidence_ok = _evidence_payload_matches_submission(payload, raw)
    math_ok = bool(isinstance(raw, dict) and mathematics(raw.get("result")))
    scope_ok = bool(
        isinstance(raw, dict)
        and raw.get("scope") == SCOPE
        and raw.get("completeness") == "COMPLETE"
        and raw.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        contract
        and isinstance(raw, dict)
        and raw.get("claimed_assurance") in {"UNVERIFIED", "COMPUTED", "CHECKED"}
    )
    values = {
        "input_binding": float(workspace_input_is_bound()),
        "protocol": float(bool(contract)),
        "mathematics": float(math_ok),
        "correctness": float(math_ok),
        "evidence": float(evidence_ok),
        "evidence_validity": float(evidence_ok),
        "scope": float(scope_ok),
        "scope_accuracy": float(scope_ok),
        "assurance": float(assurance_ok),
        "assurance_calibration": float(assurance_ok),
    }
    reward = float(all(values.values()))
    values.update(
        {
            "aggregate_reward": reward,
            "reward": reward,
            "false_certification": bool(
                isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED"
            ),
        }
    )
    _write(values)


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
                "evidence": 0.0,
                "evidence_validity": 0.0,
                "scope": 0.0,
                "scope_accuracy": 0.0,
                "assurance": 0.0,
                "assurance_calibration": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "false_certification": False,
                "error": type(exc).__name__,
            }
        )
