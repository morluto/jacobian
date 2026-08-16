from __future__ import annotations

import json
from itertools import product
from pathlib import Path

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
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
