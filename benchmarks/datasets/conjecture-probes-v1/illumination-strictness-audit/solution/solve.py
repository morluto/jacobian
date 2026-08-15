from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

TASK_ID = "jacobian/illumination-strictness-audit"
LIMITATIONS = [
    "ONE_THREE_DIMENSIONAL_CUBE",
    "VERTEX_SIGN_CONE_MODEL_ONLY",
    "GENERAL_ILLUMINATION_CONJECTURE_NOT_ASSESSED",
]
VERTICES = list(product((-1, 1), repeat=3))


def weak(v, d):
    return all(a * b <= 0 for a, b in zip(v, d, strict=True))


def strict(v, d):
    return all(a * b < 0 for a, b in zip(v, d, strict=True))


def main():
    root = Path("/app")
    flawed = [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]
    repair = [[-a for a in v] for v in VERTICES]
    false_pairs = [
        {"vertex_index": i, "direction_index": j}
        for i, v in enumerate(VERTICES)
        for j, d in enumerate(flawed)
        if weak(v, d) and not strict(v, d)
    ]
    result = {
        "flawed_directions": flawed,
        "weak_false_positive_pairs": false_pairs,
        "repair_directions": repair,
        "vertex_to_direction": [
            next(j for j, d in enumerate(repair) if strict(v, d)) for v in VERTICES
        ],
    }
    payload = {"schema_version": "1", "task_id": TASK_ID, "result": result}
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
