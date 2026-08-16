from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def rat(value) -> dict[str, int]:
    parsed = Fraction(value)
    return {"numerator": parsed.numerator, "denominator": parsed.denominator}


def pair(values) -> list[dict[str, int]]:
    return [rat(item) for item in values]


def main():
    root = Path("/app")
    result = {
        "factors": [[1, -3, 1], [1, -1, 1], [1, 1, 1], [2, -5, 2]],
        "outside_contributions": [
            pair(["3/2", "1/2"]),
            pair(["1", "0"]),
            pair(["1", "0"]),
            pair(["2", "0"]),
        ],
        "flawed_monic_result": pair(["3", "1"]),
        "leading_coefficient": rat("2"),
        "corrected_mahler_measure": pair(["6", "2"]),
    }
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
