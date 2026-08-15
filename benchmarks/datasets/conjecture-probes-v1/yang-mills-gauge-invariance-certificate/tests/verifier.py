"""Exact rational quaternion verifier for one finite SU(2) plaquette."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    resolve_evidence,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/yang-mills-gauge-invariance-certificate"


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

    # Reject fully commuting witnesses: at least one link/gauge pair
    # must have a nonzero commutator to exercise noncommutative
    # gauge-covariance.
    def _comm_nonzero(a, b):
        ab = mul(a, b)
        ba = mul(b, a)
        return ab != ba

    if not any(_comm_nonzero(links[i], gauges[i]) for i in range(4)):
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
    normalize_reward_file(p / "reward.json")


def main():
    ib = workspace_input_is_bound()
    s = load_submission(require_input_binding=False)
    protocol = isinstance(s, dict)
    result = s.get("result") if protocol else None
    m = bool(protocol and mathematics(result))
    witness = s.get("witness") if protocol else None
    descriptor = witness[0] if isinstance(witness, list) and len(witness) == 1 else None
    payload = (
        read_evidence_json(descriptor, expected_path="evidence/answer.txt")
        if resolve_evidence(descriptor, expected_path="evidence/answer.txt") is not None
        else None
    )
    witness_valid = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and json_value_equal(payload.get("result"), result)
    )
    agg = float(ib and protocol and m and witness_valid)
    reward(
        {
            "protocol": float(protocol),
            "input_binding": 1.0 if ib else 0.0,
            "mathematics": 1.0 if m else 0.0,
            "witness_validity": float(witness_valid),
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
                "witness_validity": 0.0,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
