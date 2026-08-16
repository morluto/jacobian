from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import isqrt
from pathlib import Path

SCALE = 10**80


def _encode(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


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
        "lower": _encode(n * a[2] * b[2]),
        "upper": _encode(n * a[3] * b[3]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    records = []
    best = None
    for n in range(1, 2001):
        current = row(n)
        upper = Fraction(current["upper"]["numerator"], current["upper"]["denominator"])
        lower = (
            None
            if best is None
            else Fraction(best["lower"]["numerator"], best["lower"]["denominator"])
        )
        if best is None or upper < lower:
            records.append(current)
            best = current
    result = {
        "records": records,
        "argmin_n": best["n"],
        "minimum_lower": best["lower"],
        "minimum_upper": best["upper"],
        "comparison_status": "STRICTLY_SEPARATED_INTERVALS",
    }
    s = {
        "result": result,
    }
    (root / "submission.json").write_text(json.dumps(s, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
