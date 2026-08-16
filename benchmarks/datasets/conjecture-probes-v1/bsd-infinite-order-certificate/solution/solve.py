from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


def add(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], a: int
) -> tuple[Fraction, Fraction]:
    x1, y1 = left
    x2, y2 = right
    slope = (3 * x1 * x1 + a) / (2 * y1) if left == right else (y2 - y1) / (x2 - x1)
    x3 = slope * slope - x1 - x2
    return x3, slope * (x1 - x3) - y1


def encode(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def pair(point: tuple[Fraction, Fraction]) -> list[dict[str, int]]:
    return [encode(point[0]), encode(point[1])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    p = (Fraction(3), Fraction(5))
    double = add(p, p, 0)
    triple = add(double, p, 0)
    result = {
        "A": 0,
        "B": -2,
        "point": [3, 5],
        "discriminant": -1728,
        "y_square": 25,
        "y_square_divides_discriminant": False,
        "double": pair(double),
        "triple": pair(triple),
        "order_conclusion": "INFINITE_BY_LUTZ_NAGELL",
    }
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
