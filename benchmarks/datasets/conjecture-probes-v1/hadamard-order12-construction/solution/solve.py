"""Produce a Paley order-12 Oracle certificate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TASK_ID = "jacobian/hadamard-order12-construction"
SCOPE = "hadamard-order12-construction:normalized-v1"
LIMITATIONS = ["ORDER_12_ONLY", "NO_GENERAL_HADAMARD_CONJECTURE_CONCLUSION"]


def determinant(matrix: list[list[int]]) -> int:
    work = [row[:] for row in matrix]
    sign = 1
    previous = 1
    for pivot_index in range(len(work) - 1):
        pivot_row = next(
            row for row in range(pivot_index, len(work)) if work[row][pivot_index]
        )
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = work[pivot_row], work[pivot_index]
            sign = -sign
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, len(work)):
            for column in range(pivot_index + 1, len(work)):
                work[row][column] = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                ) // previous
            work[row][pivot_index] = 0
        previous = pivot
    return sign * work[-1][-1]


def main() -> None:
    root = (
        Path(sys.argv[2])
        if len(sys.argv) == 3 and sys.argv[1] == "--root"
        else Path("/app")
    )
    modulus = 11
    residues = {value * value % modulus for value in range(1, modulus)}
    matrix = [[1] * 12]
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
    gram = [
        [sum(matrix[i][k] * matrix[j][k] for k in range(12)) for j in range(12)]
        for i in range(12)
    ]
    det = determinant(matrix)
    result = {
        "matrix": matrix,
        "gram": gram,
        "determinant": det,
        "determinant_abs": abs(det),
        "scope_identity": SCOPE,
    }
    evidence = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
        "limitations": LIMITATIONS,
    }
    evidence_path = root / "evidence/answer.txt"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
    )
    digest = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    submission = {
        "result": result,
        "witness": [{"path": "evidence/answer.txt", "sha256": digest}],
    }
    (root / "submission.json").write_text(
        json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
