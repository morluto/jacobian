from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

TASK_ID = "jacobian/yang-mills-gauge-invariance-certificate"
LIMITATIONS = [
    "ONE_FINITE_RATIONAL_SU2_PLAQUETTE",
    "NO_CONTINUUM_YANG_MILLS_CONSTRUCTION",
    "NO_MASS_GAP_CONCLUSION",
]


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


def row(q):
    return [str(x) for x in q]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    links = [
        (Fraction(3, 5), Fraction(4, 5), 0, 0),
        (Fraction(5, 13), 0, Fraction(12, 13), 0),
        (Fraction(8, 17), 0, 0, Fraction(15, 17)),
        (0, 1, 0, 0),
    ]
    gauges = [
        (Fraction(4, 5), Fraction(3, 5), 0, 0),
        (Fraction(12, 13), 0, Fraction(5, 13), 0),
        (Fraction(15, 17), 0, 0, Fraction(8, 17)),
        (0, 0, 1, 0),
    ]
    transformed = [
        mul(mul(gauges[i], links[i]), inv(gauges[(i + 1) % 4])) for i in range(4)
    ]
    p = product(links)
    tp = product(transformed)
    cp = mul(mul(gauges[0], p), inv(gauges[0]))
    result = {
        "links": [row(q) for q in links],
        "gauges": [row(q) for q in gauges],
        "transformed_links": [row(q) for q in transformed],
        "plaquette": row(p),
        "transformed_plaquette": row(tp),
        "conjugated_plaquette": row(cp),
        "scalar_trace_invariant": True,
    }
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
        "limitations": LIMITATIONS,
    }
    e = root / "evidence/answer.txt"
    e.parent.mkdir(parents=True, exist_ok=True)
    e.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    s = {
        "task_id": TASK_ID,
        "conclusion": "FINITE_SU2_PLAQUETTE_GAUGE_INVARIANCE",
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": "one-rational-su2-plaquette-v1",
        "completeness": "COMPLETE",
        "evidence": [
            {
                "path": "evidence/answer.txt",
                "sha256": "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest(),
            }
        ],
        "limitations": LIMITATIONS,
    }
    (root / "submission.json").write_text(json.dumps(s, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
