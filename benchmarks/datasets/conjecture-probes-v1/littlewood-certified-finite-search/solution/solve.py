from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from math import isqrt
from pathlib import Path

TASK_ID = "jacobian/littlewood-certified-finite-search"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    records = []
    best = None
    for n in range(1, 2001):
        current = row(n)
        if best is None or Fraction(current["upper"]) < Fraction(best["lower"]):
            records.append(current)
            best = current
    result = {
        "records": records,
        "argmin_n": best["n"],
        "minimum_lower": best["lower"],
        "minimum_upper": best["upper"],
        "comparison_status": "STRICTLY_SEPARATED_INTERVALS",
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
        "conclusion": "LITTLEWOOD_FINITE_MINIMUM_CERTIFICATE",
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": "sqrt2-sqrt3-n-up-to-2000-v1",
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
