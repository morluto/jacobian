from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def projective_triples() -> list[list[int]]:
    triples = []
    for value in itertools.product(range(3), repeat=3):
        if value == (0, 0, 0):
            continue
        first = next(x for x in value if x)
        inverse = 1 if first == 1 else 2
        normalized = tuple((inverse * x) % 3 for x in value)
        if normalized == value:
            triples.append(list(value))
    return sorted(triples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    points = projective_triples()
    lines = projective_triples()
    edges = [
        [i, j]
        for i, point in enumerate(points)
        for j, line in enumerate(lines)
        if sum(a * b for a, b in zip(point, line, strict=True)) % 3 == 0
    ]
    pair_rows = [
        {"pair": [i, j], "common_neighbors": 1}
        for i, j in itertools.combinations(range(13), 2)
    ]
    result = {
        "points": points,
        "lines": lines,
        "edges": edges,
        "left_degrees": [4] * 13,
        "right_degrees": [4] * 13,
        "left_pair_common_counts": pair_rows,
        "right_pair_common_counts": pair_rows,
        "edge_count": 52,
        "pair_budget": 78,
        "excluded_edge_count": 53,
    }
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
