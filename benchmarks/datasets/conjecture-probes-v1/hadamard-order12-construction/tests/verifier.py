"""Clean-room verifier for a normalized order-12 Hadamard construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifier_support import (
    aggregate_reward,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    witness_list_is_bound,
    workspace_input_is_bound,
)

SCOPE = "hadamard-order12-construction:normalized-v1"


def _determinant(matrix: list[list[int]]) -> int:
    work = [row[:] for row in matrix]
    sign = 1
    previous = 1
    for pivot_index in range(len(work) - 1):
        pivot_row = next(
            (row for row in range(pivot_index, len(work)) if work[row][pivot_index]),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = work[pivot_row], work[pivot_index]
            sign = -sign
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, len(work)):
            for column in range(pivot_index + 1, len(work)):
                numerator = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                )
                if numerator % previous:
                    raise ArithmeticError("non-exact Bareiss division")
                work[row][column] = numerator // previous
            work[row][pivot_index] = 0
        previous = pivot
    return sign * work[-1][-1]


def _mathematics(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "matrix",
        "gram",
        "determinant",
        "determinant_abs",
        "scope_identity",
    }:
        return False
    matrix = result.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 12:
        return False
    if any(
        not isinstance(row, list)
        or len(row) != 12
        or any(type(value) is not int or value not in {-1, 1} for value in row)
        for row in matrix
    ):
        return False
    if any(value != 1 for value in matrix[0]) or any(row[0] != 1 for row in matrix):
        return False
    gram = [
        [sum(matrix[i][k] * matrix[j][k] for k in range(12)) for j in range(12)]
        for i in range(12)
    ]
    expected_gram = [[12 if i == j else 0 for j in range(12)] for i in range(12)]
    submitted_gram = result.get("gram")
    if (
        not isinstance(submitted_gram, list)
        or len(submitted_gram) != 12
        or any(
            not isinstance(row, list)
            or len(row) != 12
            or any(type(value) is not int for value in row)
            for row in submitted_gram
        )
    ):
        return False
    if gram != expected_gram or submitted_gram != gram:
        return False
    determinant = _determinant(matrix)
    return (
        result.get("determinant") == determinant
        and type(result.get("determinant")) is int
        and result.get("determinant_abs") == abs(determinant) == 12**6
        and result.get("scope_identity") == SCOPE
    )


def _reward(value: dict[str, Any]) -> None:
    path = Path("/logs/verifier")
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward.json").write_text(json.dumps(value, sort_keys=True))
    normalize_reward_file(path / "reward.json")


def main() -> None:
    input_bound = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    protocol_ok = submission is not None
    mathematics = bool(
        protocol_ok and input_bound and _mathematics(submission.get("result"))
    )
    evidence = bool(protocol_ok and witness_list_is_bound(submission.get("witness")))
    payload = (
        read_evidence_json(
            submission["witness"][0],
            expected_path="evidence/answer.txt",
        )
        if evidence
        else None
    )
    evidence = bool(
        isinstance(payload, dict)
        and payload.get("schema_version") == "1"
        and json_value_equal(payload.get("result"), submission.get("result"))
    )
    reward = aggregate_reward(
        correctness=mathematics,
        witness_validity=evidence,
        protocol_ok=protocol_ok and input_bound,
    )
    _reward(
        {
            "protocol_compliance": float(protocol_ok),
            "input_binding": float(input_bound),
            "correctness": float(mathematics),
            "witness_validity": float(evidence),
            "reward": reward,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _reward(
            {
                "protocol_compliance": 0.0,
                "input_binding": 0.0,
                "correctness": 0.0,
                "witness_validity": 0.0,
                "reward": 0.0,
                "error": type(exc).__name__,
            }
        )
