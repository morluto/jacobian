from __future__ import annotations

import json
from pathlib import Path

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 4),
    (5, 7),
    (7, 9),
    (6, 9),
    (6, 8),
    (5, 8),
    (0, 5),
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
]
CYCLES = [
    [0, 1, 2, 3, 4],
    [0, 1, 6, 8, 5],
    [1, 2, 7, 9, 6],
    [2, 3, 8, 5, 7],
    [3, 4, 9, 6, 8],
    [0, 4, 9, 7, 5],
]


def multiplicities(cycles: list[list[int]]) -> list[int]:
    edge_index = {tuple(sorted(edge)): i for i, edge in enumerate(EDGES)}
    counts = [0] * len(EDGES)
    for cycle in cycles:
        for i, left in enumerate(cycle):
            counts[edge_index[tuple(sorted((left, cycle[(i + 1) % len(cycle)])))]] += 1
    return counts


def main() -> None:
    root = Path("/app")
    flawed = CYCLES[:-1]
    flawed_counts = multiplicities(flawed)
    result = {
        "flawed_cycles": flawed,
        "flawed_multiplicities": flawed_counts,
        "non_double_edge_indices": [
            i for i, count in enumerate(flawed_counts) if count != 2
        ],
        "repair_cycles": CYCLES,
        "repair_multiplicities": multiplicities(CYCLES),
    }
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
