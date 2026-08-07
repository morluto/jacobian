"""Clean-room verifier for a normalized order-12 Hadamard construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

TASK_ID = "jacobian/hadamard-order12-construction"
SCOPE = "hadamard-order12-construction:normalized-v1"
LIMITATIONS = ["ORDER_12_ONLY", "NO_GENERAL_HADAMARD_CONJECTURE_CONCLUSION"]


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


def _frozen() -> dict[str, Any] | None:
    try:
        value = json.loads(Path("/tests/input.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return (
        value
        if isinstance(value, dict)
        and value.get("task_id") == TASK_ID
        and value.get("order") == 12
        else None
    )


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


def main() -> None:
    input_bound = workspace_input_is_bound()
    frozen = _frozen() if input_bound else None
    submission = load_submission(require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion="HADAMARD_ORDER12_CONSTRUCTION",
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    mathematics = bool(
        frozen
        and isinstance(submission, dict)
        and _mathematics(submission.get("result"))
    )
    evidence = bool(
        isinstance(submission, dict)
        and evidence_list_is_bound(submission.get("evidence"))
    )
    payload = (
        read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/answer.txt",
        )
        if evidence
        else None
    )
    evidence = bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload.get("schema_version") == "1"
        and payload.get("task_id") == TASK_ID
        and payload.get("result") == submission.get("result")
        and payload.get("limitations") == LIMITATIONS
    )
    scope = bool(
        isinstance(submission, dict)
        and submission.get("scope") == SCOPE
        and submission.get("limitations") == LIMITATIONS
    )
    scoreable_assurances = frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"})
    assurance = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") in scoreable_assurances
    )
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    aggregate = (
        1.0
        if all((input_bound, contract, mathematics, evidence, scope, assurance))
        and not false_certification
        else 0.0
    )
    _reward(
        {
            "protocol": float(contract),
            "input_binding": float(input_bound),
            "mathematics": float(mathematics),
            "evidence": float(evidence),
            "scope": float(scope),
            "assurance": float(assurance),
            "false_certification": false_certification,
            "aggregate_reward": aggregate,
            "reward": aggregate,
            "correctness": 1.0 if mathematics else 0.0,
            "evidence_validity": 1.0 if evidence else 0.0,
            "scope_accuracy": 1.0 if scope else 0.0,
            "assurance_calibration": 1.0 if assurance else 0.0,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _reward(
            {
                "protocol": 0.0,
                "input_binding": 0.0,
                "mathematics": 0.0,
                "evidence": 0.0,
                "scope": 0.0,
                "assurance": 0.0,
                "false_certification": False,
                "aggregate_reward": 0.0,
                "reward": 0.0,
                "correctness": 0.0,
                "evidence_validity": 0.0,
                "scope_accuracy": 0.0,
                "assurance_calibration": 0.0,
                "error": type(exc).__name__,
            }
        )
