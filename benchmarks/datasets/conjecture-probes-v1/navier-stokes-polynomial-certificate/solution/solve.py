from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


def rat(value) -> dict[str, int]:
    parsed = Fraction(value)
    return {"numerator": parsed.numerator, "denominator": parsed.denominator}


TASK_ID = "jacobian/navier-stokes-polynomial-certificate"
LIMITATIONS = [
    "ONE_EXACT_2D_STEADY_POLYNOMIAL_FIELD",
    "NO_GLOBAL_NAVIER_STOKES_REGULARITY_CONCLUSION",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    result = {
        "velocity": [
            [rat(0), rat(0), rat(-1)],
            [rat(0), rat(1), rat(0)],
        ],
        "pressure": [rat(0), rat(0), rat(0), rat("1/2"), rat(0), rat("1/2")],
        "divergence": [rat(0)],
        "momentum_x": [rat(0), rat(0), rat(0)],
        "momentum_y": [rat(0), rat(0), rat(0)],
        "vorticity": rat(2),
    }
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
        "limitations": LIMITATIONS,
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
