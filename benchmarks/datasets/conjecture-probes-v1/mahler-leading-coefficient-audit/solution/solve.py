from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

TASK_ID = "jacobian/mahler-leading-coefficient-audit"
LIMITATIONS = [
    "ONE_DEGREE_EIGHT_POLYNOMIAL",
    "EXACT_FACTOR_FORMULA_AUDIT_ONLY",
    "LEHMER_PROBLEM_NOT_ASSESSED",
]


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
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
        "limitations": LIMITATIONS,
    }
    evidence = root / "evidence/answer.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission = {
        "result": result,
        "witness": [
            {
                "path": "evidence/answer.json",
                "sha256": "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
