from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


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
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
