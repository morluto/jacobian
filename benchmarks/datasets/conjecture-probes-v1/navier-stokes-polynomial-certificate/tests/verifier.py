"""Exact verifier for one affine steady incompressible flow certificate."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/navier-stokes-polynomial-certificate"
SCOPE = "steady-affine-2d-polynomial-fields-v1"
LIMITATIONS = [
    "ONE_EXACT_2D_STEADY_POLYNOMIAL_FIELD",
    "NO_GLOBAL_NAVIER_STOKES_REGULARITY_CONCLUSION",
]
MAX_EVIDENCE_BYTES = None
scoreable_assurances = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})


def _rat(value: object, num_bound: int = 50, den_bound: int = 20) -> Fraction:
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError
    parsed = Fraction(value)
    if (
        str(parsed) != value
        or abs(parsed.numerator) > num_bound
        or parsed.denominator > den_bound
    ):
        raise ValueError
    return parsed


def _vector(
    value: object, length: int, num_bound: int = 50, den_bound: int = 20
) -> list[Fraction]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError
    return [_rat(item, num_bound, den_bound) for item in value]


def _frozen_input() -> dict[str, Any] | None:
    """Read coefficient bounds from /tests/input.json instead of hard-coding."""
    try:
        value = json.loads(Path("/tests/input.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("task_id") != TASK_ID:
        return None
    return value


def _mathematics(result: Any, num_bound: int = 50, den_bound: int = 20) -> bool:
    try:
        if not isinstance(result, dict) or set(result) != {
            "velocity",
            "pressure",
            "divergence",
            "momentum_x",
            "momentum_y",
            "vorticity",
        }:
            return False
        if not isinstance(result["velocity"], list) or len(result["velocity"]) != 2:
            return False
        a = _vector(result["velocity"][0], 3, num_bound, den_bound)
        b = _vector(result["velocity"][1], 3, num_bound, den_bound)
        p = _vector(result["pressure"], 6, num_bound, den_bound)
        divergence = [a[1] + b[2]]
        momentum_x = [
            a[1] * a[0] + a[2] * b[0] + p[1],
            a[1] * a[1] + a[2] * b[1] + 2 * p[3],
            a[1] * a[2] + a[2] * b[2] + p[4],
        ]
        momentum_y = [
            b[1] * a[0] + b[2] * b[0] + p[2],
            b[1] * a[1] + b[2] * b[1] + p[4],
            b[1] * a[2] + b[2] * b[2] + 2 * p[5],
        ]
        vorticity = b[1] - a[2]
        submitted = (
            _vector(result["divergence"], 1, num_bound, den_bound),
            _vector(result["momentum_x"], 3, num_bound, den_bound),
            _vector(result["momentum_y"], 3, num_bound, den_bound),
            _rat(result["vorticity"], num_bound, den_bound),
        )
    except (ValueError, ZeroDivisionError, TypeError):
        return False
    expected = (divergence, momentum_x, momentum_y, vorticity)
    return (
        submitted == expected
        and divergence == [0]
        and momentum_x == [0, 0, 0]
        and momentum_y == [0, 0, 0]
        and vorticity != 0
    )


def _raw_submission() -> dict[str, Any] | None:
    if not is_regular_bounded_file(
        Path("/app/submission.json"), max_bytes=MAX_SUBMISSION_BYTES
    ):
        return None
    try:
        value = json.loads(Path("/app/submission.json").read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _reward(value: dict[str, Any]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def main() -> None:
    input_bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="STEADY_INCOMPRESSIBLE_POLYNOMIAL_CERTIFICATE",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    frozen = _frozen_input()
    num_bound = frozen.get("coefficient_numerator_bound", 50) if frozen else 50
    den_bound = frozen.get("coefficient_denominator_bound", 20) if frozen else 20
    mathematics = bool(
        isinstance(submission, dict)
        and _mathematics(submission.get("result"), num_bound, den_bound)
    )
    evidence = bool(
        isinstance(submission, dict)
        and evidence_list_is_bound(submission.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/answer.txt",
            max_bytes=None,
        )
        if evidence
        else None
    )
    evidence = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload
        == {
            "schema_version": "1",
            "task_id": TASK_ID,
            "result": submission.get("result"),
            "limitations": LIMITATIONS,
        }
    )
    scope = bool(
        isinstance(submission, dict)
        and submission.get("scope") == SCOPE
        and submission.get("limitations") == LIMITATIONS
    )
    assurance = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") in scoreable_assurances
    )
    raw_submission = _raw_submission()
    false_certification = bool(
        isinstance(raw_submission, dict)
        and raw_submission.get("claimed_assurance") == "VERIFIED"
    )
    aggregate = (
        1.0
        if all((input_bound, contract, mathematics, evidence, scope, assurance))
        and not false_certification
        else 0.0
    )
    _reward(
        {
            "protocol": 1.0 if contract else 0.0,
            "input_binding": 1.0 if input_bound else 0.0,
            "mathematics": 1.0 if mathematics else 0.0,
            "evidence": 1.0 if evidence else 0.0,
            "scope": 1.0 if scope else 0.0,
            "assurance": 1.0 if assurance else 0.0,
            "false_certification": false_certification,
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
                "evidence": 0.0,
                "scope": 0.0,
                "assurance": 0.0,
                "false_certification": False,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
