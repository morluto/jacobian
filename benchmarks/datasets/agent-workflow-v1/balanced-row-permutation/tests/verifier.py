from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
TASK_ID = "jacobian/balanced-row-permutation"
CONCLUSION = "BALANCED_ROW_PERMUTATION_CONSTRUCTED"
SCOPE = "the frozen 8 by 6 matrix"


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        data = json.loads(hidden)
        return bool(
            (WORKSPACE / "input.json").read_bytes() == hidden
            and data["source"]["revision"] == "882ba08befd0856f5364db1e53d58c7e2cf704f9"
            and data["source"]["row"] == 34
        )
    except (OSError, ValueError, KeyError):
        return False


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
    if (
        not isinstance(permutations, list)
        or len(permutations) != 8
        or not isinstance(output, list)
        or len(output) != 8
        or not isinstance(layers, list)
        or len(layers) != 6
    ):
        return False
    for row in range(8):
        permutation = permutations[row]
        result_row = output[row]
        if (
            not isinstance(permutation, list)
            or sorted(permutation) != list(range(6))
            or not isinstance(result_row, list)
            or len(result_row) != 6
            or result_row != [matrix[row][position] for position in permutation]
        ):
            return False
    for column, layer in enumerate(layers):
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
        if Counter(symbols) != Counter({1: 2, 2: 2, 3: 2, 4: 2}):
            return False
    global_counts = Counter(item for row in matrix for item in row)
    return global_counts == Counter({1: 12, 2: 12, 3: 12, 4: 12})


def _evidence(value: object) -> bool:
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
        text = path.read_text().casefold()
    except (OSError, UnicodeError):
        return False
    return all(word in text for word in ("column", "row", "source position", "exactly"))


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
    correct = bool(
        protocol
        and isinstance(submission, dict)
        and _source_is_bound()
        and submission.get("scope") == SCOPE
        and submission.get("completeness") == "COMPLETE"
        and submission.get("claimed_assurance") == "COMPUTED"
        and _result(submission.get("result"))
        and _evidence(submission.get("evidence"))
        and any(
            "general theorem" in item.casefold()
            for item in submission.get("limitations", [])
        )
    )
    score = 1.0 if correct and not false_certification else 0.0
    return {
        "correctness": score,
        "evidence_validity": score,
        "scope_accuracy": score,
        "assurance_calibration": score,
        "reward": score,
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
