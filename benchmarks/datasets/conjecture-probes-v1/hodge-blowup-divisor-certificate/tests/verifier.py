"""Exact polynomial and intersection verifier for one blow-up divisor."""

from __future__ import annotations

import json
from math import gcd
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/hodge-blowup-divisor-certificate"
SCOPE = "six-point-p2-blowup-divisor-v1"
LIMITATIONS = [
    "ONE_CUBIC_DIVISOR_ON_ONE_BLOWUP",
    "LEFSCHETZ_1_1_TRUSTED",
    "NO_HIGHER_CODIMENSION_HODGE_CONCLUSION",
]
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


def _raw_submission() -> dict[str, Any] | None:
    """Read raw submission JSON before strict validation for false-certification detection."""
    path = Path("/app/submission.json")
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main():
    ib = workspace_input_is_bound()
    s = load_submission(require_input_binding=False)
    c = strict_submission_contract(
        s,
        task_id=TASK_ID,
        conclusion="ALGEBRAIC_DIVISOR_CLASS_CERTIFICATE",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    raw = _raw_submission()
    m = bool(isinstance(raw, dict) and mathematics(raw.get("result")))
    e = bool(
        isinstance(s, dict)
        and evidence_list_is_bound(s.get("evidence"), max_bytes=None)
    )
    payload = (
        read_evidence_json(
            s["evidence"][0],
            expected_path="evidence/answer.txt",
            max_bytes=None,
        )
        if e
        else None
    )
    e = bool(
        isinstance(payload, dict)
        and payload
        == {
            "schema_version": "1",
            "task_id": TASK_ID,
            "result": s.get("result"),
            "limitations": LIMITATIONS,
        }
    )
    sc = bool(
        isinstance(s, dict)
        and s.get("scope") == SCOPE
        and s.get("limitations") == LIMITATIONS
    )
    scoreable_assurances = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})
    a = bool(isinstance(s, dict) and s.get("claimed_assurance") in scoreable_assurances)
    f = bool(isinstance(raw, dict) and raw.get("claimed_assurance") == "VERIFIED")
    agg = 1.0 if all((ib, c, m, e, sc, a)) and not f else 0.0
    reward(
        {
            "protocol": 1.0 if c else 0.0,
            "input_binding": 1.0 if ib else 0.0,
            "mathematics": 1.0 if m else 0.0,
            "evidence": 1.0 if e else 0.0,
            "scope": 1.0 if sc else 0.0,
            "assurance": 1.0 if a else 0.0,
            "false_certification": f,
            "aggregate_reward": agg,
            "reward": agg,
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
                "evidence": 0.0,
                "scope": 0.0,
                "assurance": 0.0,
                "false_certification": False,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
