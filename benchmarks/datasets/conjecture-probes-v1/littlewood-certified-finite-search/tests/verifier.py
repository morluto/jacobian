"""Rigorous rational-interval verifier for a finite Littlewood search."""

from __future__ import annotations

import json
from fractions import Fraction
from math import isqrt
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/littlewood-certified-finite-search"
SCOPE = "sqrt2-sqrt3-n-up-to-2000-v1"
LIMITATIONS = [
    "ONE_FIXED_QUADRATIC_IRRATIONAL_PAIR",
    "N_AT_MOST_2000",
    "NO_LIMINF_OR_LITTLEWOOD_CONCLUSION",
]
SCALE = 10**80


def bounds(d, n):
    root = isqrt(d * SCALE * SCALE)
    lo = Fraction(root, SCALE)
    hi = Fraction(root + 1, SCALE)
    floor = isqrt(d * n * n)
    if 4 * d * n * n < (2 * floor + 1) ** 2:
        return floor, floor, n * lo - floor, n * hi - floor
    return floor, floor + 1, floor + 1 - n * hi, floor + 1 - n * lo


def row(n):
    a = bounds(2, n)
    b = bounds(3, n)
    return {
        "n": n,
        "floors": [a[0], b[0]],
        "nearest": [a[1], b[1]],
        "lower": str(n * a[2] * b[2]),
        "upper": str(n * a[3] * b[3]),
    }


def expected():
    rows = []
    best = None
    for n in range(1, 2001):
        current = row(n)
        hi = Fraction(current["upper"])
        if best is None or hi < Fraction(best["lower"]):
            rows.append(current)
            best = current
    assert best is not None and all(
        Fraction(best["upper"]) < Fraction(row(n)["lower"])
        for n in range(1, 2001)
        if n != best["n"]
    )
    return rows, best


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
    return result == {
        "records": rows,
        "argmin_n": best["n"],
        "minimum_lower": best["lower"],
        "minimum_upper": best["upper"],
        "comparison_status": "STRICTLY_SEPARATED_INTERVALS",
    }


def reward(v):
    p = Path("/logs/verifier")
    p.mkdir(parents=True, exist_ok=True)
    (p / "reward.json").write_text(json.dumps(v, sort_keys=True))
    normalize_reward_file(p / "reward.json")


def main():
    ib = workspace_input_is_bound()
    s = load_submission(require_input_binding=False)
    c = strict_submission_contract(
        s,
        task_id=TASK_ID,
        conclusion="LITTLEWOOD_FINITE_MINIMUM_CERTIFICATE",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    m = bool(isinstance(s, dict) and mathematics(s.get("result")))
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
    a = bool(isinstance(s, dict) and s.get("claimed_assurance") == "CHECKED")
    f = bool(isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED")
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
