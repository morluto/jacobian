from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

TASK_ID = "jacobian/bsd-infinite-order-certificate"


def add(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], a: int
) -> tuple[Fraction, Fraction]:
    x1, y1 = left
    x2, y2 = right
    slope = (3 * x1 * x1 + a) / (2 * y1) if left == right else (y2 - y1) / (x2 - x1)
    x3 = slope * slope - x1 - x2
    return x3, slope * (x1 - x3) - y1


def pair(point: tuple[Fraction, Fraction]) -> list[str]:
    return [str(point[0]), str(point[1])]


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
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
    }
    evidence = root / "evidence/answer.txt"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission = {
        "result": result,
        "witness": [
            {
                "path": "evidence/answer.txt",
                "sha256": "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
