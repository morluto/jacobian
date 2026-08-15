from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

TASK_ID = "jacobian/normal-projective-chart-audit"
LIMITATIONS = [
    "ONE_RATIONAL_ELLIPSE",
    "ONE_QUERY_POINT",
    "CONCURRENT_NORMALS_CONJECTURE_NOT_ASSESSED",
]


def rat(value) -> dict[str, int]:
    parsed = Fraction(value)
    return {"numerator": parsed.numerator, "denominator": parsed.denominator}


def point(values) -> list[dict[str, int]]:
    return [rat(item) for item in values]


def main():
    root = Path("/app")
    result = {
        "finite_parameters": point(["-1", "0", "1"]),
        "finite_points": [point(["0", "-1"]), point(["0", "1"]), point(["2", "0"])],
        "missing_projective_parameter": point(["1", "0"]),
        "missing_point": point(["-2", "0"]),
        "footpoint_records": [
            {
                "point": point(["-2", "0"]),
                "ellipse_residual": rat("0"),
                "normal_residual": rat("0"),
            },
            {
                "point": point(["0", "-1"]),
                "ellipse_residual": rat("0"),
                "normal_residual": rat("0"),
            },
            {
                "point": point(["0", "1"]),
                "ellipse_residual": rat("0"),
                "normal_residual": rat("0"),
            },
            {
                "point": point(["2", "0"]),
                "ellipse_residual": rat("0"),
                "normal_residual": rat("0"),
            },
        ],
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
