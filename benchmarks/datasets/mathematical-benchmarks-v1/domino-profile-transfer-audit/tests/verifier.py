import json
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

TESTS = Path("/tests")
MODULUS = 19
WIDTH = 2021
SIZE = 8
MAX_EVIDENCE_BYTES = 64 * 1024
SET_BITS = [bit for bit in range(WIDTH.bit_length()) if WIDTH & 1 << bit]


def _outgoing_masks(incoming: int) -> list[int]:
    outputs: list[int] = []

    def fill(occupied: int, outgoing: int) -> None:
        if occupied == 7:
            outputs.append(outgoing)
            return
        row = next(index for index in range(3) if not occupied & 1 << index)
        fill(occupied | 1 << row, outgoing | 1 << row)
        if row < 2 and (not occupied & 1 << row + 1):
            fill(occupied | 1 << row | 1 << row + 1, outgoing)

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
    if row not in {0, 2} or matrix != _transition() or (not _valid_vector(initial)):
        return False
    expected_initial = [0] * SIZE
    expected_initial[1 << row] = 1
    if initial != expected_initial or not isinstance(trace, list) or len(trace) != 8:
        return False
    vector = initial
    power = matrix
    trace_index = 0
    for bit in range(WIDTH.bit_length()):
        if WIDTH & 1 << bit:
            item = trace[trace_index]
            if (
                not isinstance(item, dict)
                or set(item) != {"bit", "before", "after"}
                or item["bit"] != bit
                or (item["before"] != vector)
                or (not _valid_vector(item["after"]))
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
        and (result["remainder"] != 4)
        and (
            result["diagnosis"]
            == "PROPOSED_DECOMPOSITION_DOES_NOT_PARTITION_ALL_TILINGS"
        )
    )


def main() -> None:
    submission = load_submission()
    result = submission.get("result") if isinstance(submission, dict) else None
    mathematical = _result_valid(result)
    reward = float(mathematical)
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps({"correctness": float(mathematical), "reward": reward})
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
