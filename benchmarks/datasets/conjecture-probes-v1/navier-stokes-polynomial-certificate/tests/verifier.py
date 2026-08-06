"""Exact verifier for one affine steady incompressible flow certificate."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
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
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024


def _rat(value: object) -> Fraction:
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError
    parsed = Fraction(value)
    if str(parsed) != value or abs(parsed.numerator) > 50 or parsed.denominator > 20:
        raise ValueError
    return parsed


def _vector(value: object, length: int) -> list[Fraction]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError
    return [_rat(item) for item in value]


def _mathematics(result: Any) -> bool:
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
        a = _vector(result["velocity"][0], 3)
        b = _vector(result["velocity"][1], 3)
        p = _vector(result["pressure"], 6)
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
            _vector(result["divergence"], 1),
            _vector(result["momentum_x"], 3),
            _vector(result["momentum_y"], 3),
            _rat(result["vorticity"]),
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


def _reward(value: dict[str, Any]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))


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
    mathematics = bool(contract and _mathematics(submission["result"]))
    evidence = bool(
        contract
        and evidence_list_is_bound(submission["evidence"], max_bytes=MAX_EVIDENCE_BYTES)
    )
    payload = (
        read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/answer.txt",
            max_bytes=MAX_EVIDENCE_BYTES,
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
        contract
        and submission.get("scope") == SCOPE
        and submission.get("limitations") == LIMITATIONS
    )
    assurance = bool(contract and submission.get("claimed_assurance") == "CHECKED")
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
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
        _reward({"aggregate_reward": 0.0, "reward": 0.0, "error": type(exc).__name__})
