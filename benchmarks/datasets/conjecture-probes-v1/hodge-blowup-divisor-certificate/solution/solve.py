from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASK_ID = "jacobian/hodge-blowup-divisor-certificate"
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


def ev(c, p):
    return sum(
        v * p[0] ** a * p[1] ** b * p[2] ** d
        for v, (a, b, d) in zip(c, EXP, strict=True)
    )


def grad(c, p):
    out = []
    for axis in range(3):
        total = 0
        for v, e in zip(c, EXP, strict=True):
            if e[axis]:
                q = list(e)
                q[axis] -= 1
                total += v * e[axis] * p[0] ** q[0] * p[1] ** q[1] * p[2] ** q[2]
        out.append(total)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    c = [1, 0, -3, 0, 0, 2, 1, -3, 2, 0]
    result = {
        "coefficients": c,
        "point_checks": [
            {
                "point_index": i,
                "value": ev(c, p),
                "gradient": grad(c, p),
                "multiplicity": 1,
            }
            for i, p in enumerate(POINTS)
        ],
        "divisor_class": [3, -1, -1, -1, -1, -1, -1],
        "self_intersection": 3,
        "canonical_intersection": -3,
        "arithmetic_genus": 1,
        "cycle_classification": "ALGEBRAIC_DIVISOR_HODGE_1_1",
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
        "conclusion": "ALGEBRAIC_DIVISOR_CLASS_CERTIFICATE",
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": "six-point-p2-blowup-divisor-v1",
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
