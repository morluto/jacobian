from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    s = {
        "result": result,
    }
    (root / "submission.json").write_text(json.dumps(s, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
