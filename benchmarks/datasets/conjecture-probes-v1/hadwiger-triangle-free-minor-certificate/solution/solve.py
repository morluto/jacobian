from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASK_ID = "jacobian/hadwiger-triangle-free-minor-certificate"
LIMITATIONS = [
    "ONE_TRIANGLE_FREE_11_VERTEX_GRAPH",
    "EXHAUSTIVE_THREE_COLOR_REJECTION",
    "NO_GLOBAL_HADWIGER_CONCLUSION",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    edges = [
        [0, 1],
        [0, 4],
        [0, 6],
        [0, 9],
        [1, 2],
        [1, 5],
        [1, 7],
        [2, 3],
        [2, 6],
        [2, 8],
        [3, 4],
        [3, 7],
        [3, 9],
        [4, 5],
        [4, 8],
        [5, 10],
        [6, 10],
        [7, 10],
        [8, 10],
        [9, 10],
    ]
    result = {
        "edges": edges,
        "four_coloring": [0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3],
        "branch_sets": [[0], [1], [2, 6], [3, 4, 5]],
        "chromatic_number": 4,
        "minor_order": 4,
    }
    payload = {"schema_version": "1", "task_id": TASK_ID, "result": result}
    e = root / "evidence/answer.txt"
    e.parent.mkdir(parents=True, exist_ok=True)
    e.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    s = {
        "result": result,
        "witness": [
            {
                "path": "evidence/answer.txt",
                "sha256": "sha256:" + hashlib.sha256(e.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "submission.json").write_text(json.dumps(s, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
