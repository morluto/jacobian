"""Produce a Paley-tensor Hadamard matrix of order 664."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def paley_order_332() -> list[list[int]]:
    modulus = 331
    residues = {value * value % modulus for value in range(1, modulus)}
    matrix = [[1] * 332]
    for row in range(modulus):
        matrix.append(
            [1]
            + [
                -1
                if row == column
                else (1 if (row - column) % modulus in residues else -1)
                for column in range(modulus)
            ]
        )
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    args = parser.parse_args()
    base = paley_order_332()
    rows = []
    for row in base:
        bits = "".join("1" if value > 0 else "0" for value in row)
        complement = "".join("0" if value > 0 else "1" for value in row)
        rows.append(bits + bits)
        rows.append(bits + complement)
    submission = {"result": {"rows": rows}}
    (args.root / "submission.json").write_text(
        json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
