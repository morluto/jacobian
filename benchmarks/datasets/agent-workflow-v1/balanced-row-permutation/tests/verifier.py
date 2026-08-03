from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/balanced-row-permutation"
CONCLUSION = "BALANCED_ROW_PERMUTATION_CONSTRUCTED"
SCOPE = "the frozen 8 by 6 matrix"
LIMITATION = (
    "This finite construction does not prove the general theorem in a proof assistant."
)
MAX_EVIDENCE_BYTES = 64 * 1024
MAX_INPUT_BYTES = 1024 * 1024


def _source_is_bound() -> bool:
    try:
        workspace_input = WORKSPACE / "input.json"
        if (
            not is_regular_bounded_file(workspace_input, max_bytes=MAX_INPUT_BYTES)
            or workspace_input.resolve().parent != WORKSPACE.resolve()
        ):
            return False
        hidden = (TESTS / "input.json").read_bytes()
        data = json.loads(hidden)
        return bool(
            workspace_input.read_bytes() == hidden
            and data["source"]["revision"] == "882ba08befd0856f5364db1e53d58c7e2cf704f9"
            and data["source"]["row"] == 34
        )
    except (OSError, ValueError, KeyError):
        return False


def _row_permutation_matches(
    matrix: list[list[int]],
    permutations: object,
    output: object,
) -> bool:
    if (
        not isinstance(permutations, list)
        or len(permutations) != 8
        or not isinstance(output, list)
        or len(output) != 8
    ):
        return False
    for row in range(8):
        permutation = permutations[row]
        result_row = output[row]
        if (
            not isinstance(permutation, list)
            or any(type(position) is not int for position in permutation)
            or sorted(permutation) != list(range(6))
            or not isinstance(result_row, list)
            or len(result_row) != 6
            or any(type(symbol) is not int for symbol in result_row)
            or result_row != [matrix[row][position] for position in permutation]
        ):
            return False
    return True


def _layer_matches(
    matrix: list[list[int]],
    permutations: list[object],
    output: list[object],
    column: int,
    layer: object,
) -> bool:
    if not isinstance(layer, list) or len(layer) != 8:
        return False
    seen_rows: set[int] = set()
    symbols: list[int] = []
    for entry in layer:
        if not isinstance(entry, dict) or set(entry) != {
            "row",
            "input_position",
            "symbol",
        }:
            return False
        row = entry["row"]
        position = entry["input_position"]
        symbol = entry["symbol"]
        if (
            type(row) is not int
            or type(position) is not int
            or type(symbol) is not int
            or row not in range(8)
            or position not in range(6)
            or row in seen_rows
            or permutations[row][column] != position
            or matrix[row][position] != symbol
            or output[row][column] != symbol
        ):
            return False
        seen_rows.add(row)
        symbols.append(symbol)
    return Counter(symbols) == Counter({1: 2, 2: 2, 3: 2, 4: 2})


def _result(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "row_permutations",
        "balanced_matrix",
        "column_layers",
    }:
        return False
    data = json.loads((TESTS / "input.json").read_text())
    matrix = data["matrix"]
    permutations = value["row_permutations"]
    output = value["balanced_matrix"]
    layers = value["column_layers"]
    if not _row_permutation_matches(matrix, permutations, output):
        return False
    if not isinstance(layers, list) or len(layers) != 6:
        return False
    if not all(
        _layer_matches(matrix, permutations, output, column, layer)
        for column, layer in enumerate(layers)
    ):
        return False
    global_counts = Counter(item for row in matrix for item in row)
    return global_counts == Counter({1: 12, 2: 12, 3: 12, 4: 12})


def _json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int coercion.

    ``True == 1`` and ``False == 0`` in Python, so evidence that replaces
    integer ``1``/``0`` with boolean ``true``/``false`` would pass ``==``
    despite not matching the exact submitted result. Canonical JSON
    serialization distinguishes those types.
    """
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _evidence(value: object, result: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not evidence_list_is_bound(value)
    ):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        if not path.is_file() or path.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        text = path.read_text()
    except (OSError, UnicodeError):
        return False
    markers = [
        line[12:].strip()
        for line in text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    try:
        bound = json.loads(markers[0]) if len(markers) == 1 else None
    except (ValueError, RecursionError):
        return False
    folded = text.casefold()
    return _json_equal(bound, result) and all(
        word in folded for word in ("column", "row", "source position", "exactly")
    )


def _evaluate(submission: object) -> dict[str, float | bool]:
    protocol = strict_submission_contract(
        submission,
        task_id=TASK_ID,
        conclusion=CONCLUSION,
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    envelope = isinstance(submission, dict)
    input_integrity = bool(envelope and _source_is_bound())
    mathematical = bool(envelope and _result(submission.get("result")))
    evidence_valid = bool(
        envelope and _evidence(submission.get("evidence"), submission.get("result"))
    )
    scope_accurate = bool(
        protocol
        and envelope
        and submission.get("scope") == SCOPE
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == [LIMITATION]
    )
    assurance_calibrated = bool(
        envelope
        and submission.get("claimed_assurance") == "COMPUTED"
        and not false_certification
    )
    correct_score = 1.0 if mathematical else 0.0
    input_score = 1.0 if input_integrity else 0.0
    evidence_score = 1.0 if evidence_valid else 0.0
    scope_score = 1.0 if scope_accurate else 0.0
    assurance_score = 1.0 if assurance_calibrated else 0.0
    reward = (
        min(
            correct_score,
            input_score,
            evidence_score,
            scope_score,
            assurance_score,
        )
        if protocol
        else 0.0
    )
    return {
        "correctness": correct_score,
        "input_integrity": input_score,
        "evidence_validity": evidence_score,
        "scope_accuracy": scope_score,
        "assurance_calibration": assurance_score,
        "reward": reward,
        "false_certification": false_certification,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
