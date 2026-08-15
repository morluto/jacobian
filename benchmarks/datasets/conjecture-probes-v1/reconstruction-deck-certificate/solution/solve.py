from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASK_ID = "jacobian/reconstruction-deck-certificate"
LIMITATIONS = [
    "ONE_SCRAMBLED_NINE_CARD_DECK",
    "EXACT_CARD_EMBEDDINGS",
    "NO_GLOBAL_RECONSTRUCTION_CONCLUSION",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    original = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 8],
        [1, 2],
        [1, 4],
        [2, 3],
        [2, 5],
        [3, 4],
        [3, 7],
        [4, 5],
        [5, 6],
        [5, 8],
        [6, 7],
        [7, 8],
    ]
    maps = [
        [1, 4, 7, 2, 5, 8, 3, 6],
        [2, 5, 8, 3, 6, 0, 4, 7],
        [3, 6, 0, 4, 7, 1, 5, 8],
        [4, 7, 1, 5, 8, 2, 6, 0],
        [5, 8, 2, 6, 0, 3, 7, 1],
        [6, 0, 3, 7, 1, 4, 8, 2],
        [7, 1, 4, 8, 2, 5, 0, 3],
        [8, 2, 5, 0, 3, 6, 1, 4],
        [0, 3, 6, 1, 4, 7, 2, 5],
    ]
    result = {
        "original_edges": original,
        "embeddings": [
            {"card_id": f"card-{d}", "deleted_vertex": d, "local_to_original": maps[d]}
            for d in range(9)
        ],
        "edge_card_multiplicity": 7,
        "reconstruction_status": "EXACT_UP_TO_RELABELING",
    }
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
    }
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
