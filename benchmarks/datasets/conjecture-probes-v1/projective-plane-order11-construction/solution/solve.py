"""Produce the projective plane PG(2,11) as a line incidence list."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

MODULUS = 11


def normalized_projective_triples() -> list[tuple[int, int, int]]:
    triples: set[tuple[int, int, int]] = set()
    for value in itertools.product(range(MODULUS), repeat=3):
        if value == (0, 0, 0):
            continue
        first = next(coordinate for coordinate in value if coordinate)
        inverse = pow(first, -1, MODULUS)
        triples.add(tuple(inverse * coordinate % MODULUS for coordinate in value))
    return sorted(triples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    args = parser.parse_args()
    points = normalized_projective_triples()
    line_vectors = normalized_projective_triples()
    lines = [
        [
            index
            for index, point in enumerate(points)
            if sum(a * b for a, b in zip(point, line, strict=True)) % MODULUS == 0
        ]
        for line in line_vectors
    ]
    submission = {"result": {"lines": lines}}
    (args.root / "submission.json").write_text(
        json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
