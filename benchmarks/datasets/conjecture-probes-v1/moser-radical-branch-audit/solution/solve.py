from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

X = [
    (Fraction(1, 2), 0),
    (Fraction(-1, 2), 0),
    (Fraction(-1, 4), Fraction(-1, 12)),
    (0, 0),
    (Fraction(1, 4), Fraction(1, 12)),
    (Fraction(-1, 4), Fraction(1, 12)),
    (Fraction(1, 4), Fraction(-1, 12)),
]
Y = ["0", "0", "A", "T", "A", "B", "B"]
CLAIMED = {
    (0, 1),
    (0, 4),
    (0, 6),
    (1, 2),
    (1, 5),
    (2, 3),
    (2, 5),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 6),
}


def add(x, y):
    return x[0] + y[0], x[1] + y[1]


def square(x):
    return x[0] * x[0] + 33 * x[1] * x[1], 2 * x[0] * x[1]


def y_square(a: str, b: str, corrupted: bool):
    sa = -1 if corrupted and a == "B5" else 1
    sb = -1 if corrupted and b == "B5" else 1
    a, b = a.rstrip("5"), b.rstrip("5")
    if a == b:
        if sa == sb:
            return Fraction(0), Fraction(0)
        return {
            "A": (Fraction(17, 6), Fraction(1, 6)),
            "B": (Fraction(17, 6), Fraction(-1, 6)),
        }[a]
    key = tuple(sorted((a, b)))
    if "0" in key:
        return {
            ("0", "A"): (Fraction(17, 24), Fraction(1, 24)),
            ("0", "B"): (Fraction(17, 24), Fraction(-1, 24)),
            ("0", "T"): (Fraction(11, 4), Fraction(0)),
        }[key]
    difference = {
        ("A", "B"): (Fraction(1, 12), Fraction(0)),
        ("A", "T"): (Fraction(17, 24), Fraction(-1, 24)),
        ("B", "T"): (Fraction(17, 24), Fraction(1, 24)),
    }
    total = {
        ("A", "B"): (Fraction(11, 4), Fraction(0)),
        ("A", "T"): (Fraction(149, 24), Fraction(1, 8)),
        ("B", "T"): (Fraction(149, 24), Fraction(-1, 8)),
    }
    return total[key] if sa * sb < 0 else difference[key]


def table(corrupted: bool):
    tags = ["0", "0", "A", "T", "A", "B5", "B"]
    rows = []
    for i in range(7):
        for j in range(i + 1, 7):
            dx = X[i][0] - X[j][0], X[i][1] - X[j][1]
            a, b = add(square(dx), y_square(tags[i], tags[j], corrupted))
            rows.append(
                {
                    "pair": [i, j],
                    "distance_squared": [
                        {"numerator": a.numerator, "denominator": a.denominator},
                        {"numerator": b.numerator, "denominator": b.denominator},
                    ],
                    "unit": a == 1 and b == 0,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    corrupt, fixed = table(True), table(False)
    result = {
        "corrupted_pair_table": corrupt,
        "false_claimed_edges": [
            row["pair"]
            for row in corrupt
            if tuple(row["pair"]) in CLAIMED and not row["unit"]
        ],
        "repair": "FLIP_VERTEX_5_B_BRANCH_TO_POSITIVE",
        "corrected_pair_table": fixed,
        "corrected_edges": [row["pair"] for row in fixed if row["unit"]],
    }
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
