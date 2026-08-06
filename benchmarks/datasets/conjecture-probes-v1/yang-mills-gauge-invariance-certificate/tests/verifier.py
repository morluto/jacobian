"""Exact rational quaternion verifier for one finite SU(2) plaquette."""

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

TASK_ID = "jacobian/yang-mills-gauge-invariance-certificate"
SCOPE = "one-rational-su2-plaquette-v1"
LIMITATIONS = [
    "ONE_FINITE_RATIONAL_SU2_PLAQUETTE",
    "NO_CONTINUUM_YANG_MILLS_CONSTRUCTION",
    "NO_MASS_GAP_CONCLUSION",
]
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024


def rat(v: object, *, bounded: bool = False) -> Fraction:
    if not isinstance(v, str) or len(v) > 32:
        raise ValueError
    q = Fraction(v)
    if str(q) != v or (bounded and (abs(q.numerator) > 20 or q.denominator > 20)):
        raise ValueError
    return q


def quat(v: object, *, bounded: bool = False) -> tuple[Fraction, ...]:
    if not isinstance(v, list) or len(v) != 4:
        raise ValueError
    return tuple(rat(x, bounded=bounded) for x in v)


def mul(a, b):
    w, x, y, z = a
    bw, bx, by, bz = b
    return (
        w * bw - x * bx - y * by - z * bz,
        w * bx + x * bw + y * bz - z * by,
        w * by - x * bz + y * bw + z * bx,
        w * bz + x * by - y * bx + z * bw,
    )


def inv(a):
    return (a[0], -a[1], -a[2], -a[3])


def product(items):
    out = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
    for item in items:
        out = mul(out, item)
    return out


def mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "links",
        "gauges",
        "transformed_links",
        "plaquette",
        "transformed_plaquette",
        "conjugated_plaquette",
        "scalar_trace_invariant",
    }:
        return False
    try:
        links = [quat(v, bounded=True) for v in result["links"]]
        gauges = [quat(v, bounded=True) for v in result["gauges"]]
        transformed = [quat(v) for v in result["transformed_links"]]
        p = quat(result["plaquette"])
        tp = quat(result["transformed_plaquette"])
        cp = quat(result["conjugated_plaquette"])
    except (ValueError, TypeError, ZeroDivisionError):
        return False
    all_q = links + gauges
    if (
        len(links) != 4
        or len(gauges) != 4
        or len(transformed) != 4
        or any(sum(c * c for c in q) != 1 or q[1:] == (0, 0, 0) for q in all_q)
        or len(set(gauges)) != 4
    ):
        return False
    expected = [
        mul(mul(gauges[i], links[i]), inv(gauges[(i + 1) % 4])) for i in range(4)
    ]
    ep = product(links)
    etp = product(expected)
    ecp = mul(mul(gauges[0], ep), inv(gauges[0]))
    return (
        transformed == expected
        and p == ep
        and tp == etp
        and cp == ecp
        and etp == ecp
        and tp[0] == p[0]
        and result["scalar_trace_invariant"] is True
    )


def reward(v):
    p = Path("/logs/verifier")
    p.mkdir(parents=True, exist_ok=True)
    (p / "reward.json").write_text(json.dumps(v, sort_keys=True))


def main():
    ib = workspace_input_is_bound()
    s = load_submission(require_input_binding=False)
    c = strict_submission_contract(
        s,
        task_id=TASK_ID,
        conclusion="FINITE_SU2_PLAQUETTE_GAUGE_INVARIANCE",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    m = bool(c and mathematics(s["result"]))
    e = bool(c and evidence_list_is_bound(s["evidence"], max_bytes=MAX_EVIDENCE_BYTES))
    payload = (
        read_evidence_json(
            s["evidence"][0],
            expected_path="evidence/answer.txt",
            max_bytes=MAX_EVIDENCE_BYTES,
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
    sc = bool(c and s.get("scope") == SCOPE and s.get("limitations") == LIMITATIONS)
    a = bool(c and s.get("claimed_assurance") == "CHECKED")
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
        reward({"aggregate_reward": 0.0, "reward": 0.0, "error": type(exc).__name__})
