from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
        "velocity": [["0", "0", "-1"], ["0", "1", "0"]],
        "pressure": ["0", "0", "0", "1/2", "0", "1/2"],
        "divergence": ["0"],
        "momentum_x": ["0", "0", "0"],
        "momentum_y": ["0", "0", "0"],
        "vorticity": "2",
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
        "task_id": TASK_ID,
        "conclusion": "STEADY_INCOMPRESSIBLE_POLYNOMIAL_CERTIFICATE",
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": "steady-affine-2d-polynomial-fields-v1",
        "completeness": "COMPLETE",
        "evidence": [
            {
                "path": "evidence/answer.txt",
                "sha256": "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
        "limitations": LIMITATIONS,
    }
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
