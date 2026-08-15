"""Exact polynomial and intersection verifier for one blow-up divisor."""

from __future__ import annotations

import json
from math import gcd
from pathlib import Path
from typing import Any

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    resolve_evidence,
    witness_list_is_bound,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/hodge-blowup-divisor-certificate"
EXP = [
    (3, 0, 0),
    (2, 1, 0),
    (2, 0, 1),
    (1, 2, 0),
    (1, 1, 1),
    (1, 0, 2),
    (0, 3, 0),
    (0, 2, 1),
    (0, 1, 2),
    (0, 0, 3),
]
POINTS = [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 0, 1), (0, 2, 1)]


def evaluate(coeffs, p):
    return sum(
        c * p[0] ** a * p[1] ** b * p[2] ** d
        for c, (a, b, d) in zip(coeffs, EXP, strict=True)
    )


def gradient(coeffs, p):
    out = []
    for axis in range(3):
        total = 0
        for c, e in zip(coeffs, EXP, strict=True):
            power = e[axis]
            if power:
                ex = list(e)
                ex[axis] -= 1
                total += c * power * p[0] ** ex[0] * p[1] ** ex[1] * p[2] ** ex[2]
        out.append(total)
    return out


def mathematics(r: Any) -> bool:
    if not isinstance(r, dict) or set(r) != {
        "coefficients",
        "point_checks",
        "divisor_class",
        "self_intersection",
        "canonical_intersection",
        "arithmetic_genus",
        "cycle_classification",
    }:
        return False
    coeffs = r.get("coefficients")
    checks = r.get("point_checks")
    if (
        not isinstance(coeffs, list)
        or len(coeffs) != 10
        or any(type(c) is not int or not -20 <= c <= 20 for c in coeffs)
        or not any(coeffs)
        or gcd(*coeffs) != 1
        or not isinstance(checks, list)
        or len(checks) != 6
    ):
        return False
    expected = {
        i: {
            "point_index": i,
            "value": evaluate(coeffs, p),
            "gradient": gradient(coeffs, p),
            "multiplicity": 1,
        }
        for i, p in enumerate(POINTS)
    }
    submitted = {}
    for check in checks:
        if not isinstance(check, dict) or check.get("point_index") in submitted:
            return False
        submitted[check.get("point_index")] = check
    if submitted != expected or any(
        row["value"] != 0 or row["gradient"] == [0, 0, 0] for row in expected.values()
    ):
        return False
    d = [3, -1, -1, -1, -1, -1, -1]
    self_i = d[0] ** 2 - sum(x * x for x in d[1:])
    canonical = -3 * d[0] - sum(d[1:])
    genus = (self_i + canonical) // 2 + 1
    return (
        r["divisor_class"] == d
        and r["self_intersection"] == self_i == 3
        and r["canonical_intersection"] == canonical == -3
        and r["arithmetic_genus"] == genus == 1
        and r["cycle_classification"] == "ALGEBRAIC_DIVISOR_HODGE_1_1"
    )


def reward(v):
    p = Path("/logs/verifier")
    p.mkdir(parents=True, exist_ok=True)
    (p / "reward.json").write_text(json.dumps(v, sort_keys=True))
    normalize_reward_file(p / "reward.json")


def _witness_matches_result(witness: object, result: object) -> bool:
    if not witness_list_is_bound(witness, expected_path="evidence/answer.txt"):
        return False
    if resolve_evidence(witness[0], expected_path="evidence/answer.txt") is None:
        return False
    payload = read_evidence_json(witness[0], expected_path="evidence/answer.txt")
    return bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and json_value_equal(payload.get("result"), result)
    )


def main():
    input_bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol = isinstance(submission, dict)
    mathematics_ok = bool(protocol and mathematics(submission.get("result")))
    witness_ok = bool(
        protocol
        and _witness_matches_result(submission.get("witness"), submission.get("result"))
    )
    aggregate = float(input_bound and protocol and mathematics_ok and witness_ok)
    reward(
        {
            "protocol": float(protocol),
            "input_binding": float(input_bound),
            "mathematics": float(mathematics_ok),
            "witness_validity": float(witness_ok),
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
                "witness_validity": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
