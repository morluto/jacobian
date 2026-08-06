import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

TESTS = Path("/tests")
MODULUS = 19
WIDTH = 2021
SIZE = 8
SET_BITS = [bit for bit in range(WIDTH.bit_length()) if WIDTH & (1 << bit)]


def _outgoing_masks(incoming: int) -> list[int]:
    outputs: list[int] = []

    def fill(occupied: int, outgoing: int) -> None:
        if occupied == 0b111:
            outputs.append(outgoing)
            return
        row = next(index for index in range(3) if not occupied & (1 << index))
        # A horizontal domino crosses into the next column.
        fill(occupied | (1 << row), outgoing | (1 << row))
        # A vertical domino covers adjacent free cells in this column.
        if row < 2 and not occupied & (1 << (row + 1)):
            fill(occupied | (1 << row) | (1 << (row + 1)), outgoing)

    fill(incoming, 0)
    return outputs


def _transition() -> list[list[int]]:
    matrix = [[0] * SIZE for _ in range(SIZE)]
    for incoming in range(SIZE):
        for outgoing in _outgoing_masks(incoming):
            matrix[incoming][outgoing] += 1
    return matrix


def _matrix_multiply(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [
        [
            sum(left[row][middle] * right[middle][column] for middle in range(SIZE))
            % MODULUS
            for column in range(SIZE)
        ]
        for row in range(SIZE)
    ]


def _vector_multiply(vector: list[int], matrix: list[list[int]]) -> list[int]:
    return [
        sum(vector[middle] * matrix[middle][column] for middle in range(SIZE)) % MODULUS
        for column in range(SIZE)
    ]


def _valid_vector(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == SIZE
        and all(type(item) is int and 0 <= item < MODULUS for item in value)
    )


def _evidence_valid(value: object) -> bool:
    if not evidence_list_is_bound(value):
        return False
    target = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text().lower()
    except (OSError, UnicodeError):
        return False
    return len(text.split()) >= 45 and all(
        term in text for term in ("profile", "transition", "binary", "remainder")
    )


def _result_valid(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "removed_row",
        "transition_matrix",
        "initial_vector",
        "exponentiation_trace",
        "remainder",
        "diagnosis",
    }:
        return False
    row = result["removed_row"]
    matrix = result["transition_matrix"]
    initial = result["initial_vector"]
    trace = result["exponentiation_trace"]
    if row not in {0, 2} or matrix != _transition() or not _valid_vector(initial):
        return False
    expected_initial = [0] * SIZE
    expected_initial[1 << row] = 1
    if initial != expected_initial or not isinstance(trace, list) or len(trace) != 8:
        return False

    vector = initial
    power = matrix
    trace_index = 0
    for bit in range(WIDTH.bit_length()):
        if WIDTH & (1 << bit):
            item = trace[trace_index]
            if (
                not isinstance(item, dict)
                or set(item) != {"bit", "before", "after"}
                or item["bit"] != bit
                or item["before"] != vector
                or not _valid_vector(item["after"])
            ):
                return False
            vector = _vector_multiply(vector, power)
            if item["after"] != vector:
                return False
            trace_index += 1
        power = _matrix_multiply(power, power)

    return bool(
        [item["bit"] for item in trace] == SET_BITS
        and result["remainder"] == vector[0] == 1
        and result["remainder"] != 4
        and result["diagnosis"]
        == "PROPOSED_DECOMPOSITION_DOES_NOT_PARTITION_ALL_TILINGS"
    )


def main() -> None:
    submission = load_submission()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    mathematical = _result_valid(result)
    evidence = bool(contract and _evidence_valid(submission.get("evidence")))
    scope = bool(contract and submission.get("scope") == expected["required_scope"])
    assurance = bool(contract and submission.get("claimed_assurance") == "COMPUTED")
    false = false_verified_claim(submission, verification_record_bound=False)
    correct = bool(contract and mathematical and not false)
    reward = (
        0.0 if not correct else 0.7 + 0.1 * evidence + 0.1 * scope + 0.1 * assurance
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(mathematical),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
