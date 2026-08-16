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


def balances(flow: list[int]) -> list[int]:
    result = [0] * 10
    for value, (source, target) in zip(flow, EDGES, strict=True):
        result[source] = (result[source] + value) % 5
        result[target] = (result[target] - value) % 5
    return result


def main() -> None:
    root = Path("/app")
    flawed = [3, 2, 1, 4, 0, 1, 2, 4, 2, 1, 2, 1, 1, 2, 4]
    repair = [2, 1, 4, 3, 1, 1, 3, 3, 3, 1, 2, 1, 2, 1, 4]
    result = {
        "flawed_flow": flawed,
        "flawed_balances": balances(flawed),
        "zero_edge_index": flawed.index(0),
        "repair_flow": repair,
        "repair_balances": balances(repair),
    }
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
