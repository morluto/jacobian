from __future__ import annotations

import json
from fractions import Fraction
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
TASK_ID = "jacobian/cyclic-vector-inequality"
CONCLUSION = "SHARP_CONSTANT_CERTIFIED"
SCOPE = "all real cyclic sequences of length n > 1"


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        return bool(
            (WORKSPACE / "input.json").read_bytes() == hidden
            and json.loads(hidden)["source"]
            == {
                "dataset": "AI4Math/IneqMath",
                "revision": "3c7c32c786eb77117f3476d7f6d9af8419fa6ecc",
                "split": "dev",
                "row": 12,
                "data_id": "12",
                "license": "CC-BY-SA-4.0",
            }
        )
    except (OSError, ValueError, KeyError):
        return False


def _vectors(value: object, n: int) -> bool:
    if not isinstance(value, list) or len(value) != n:
        return False
    seen: set[int] = set()
    for vector in value:
        if not isinstance(vector, dict) or set(vector) != {
            "index",
            "first_variable",
            "second_constant",
            "second_variable",
        }:
            return False
        index = vector["index"]
        if type(index) is not int or not 1 <= index <= n or index in seen:
            return False
        seen.add(index)
        if vector != {
            "index": index,
            "first_variable": index,
            "second_constant": 1,
            "second_variable": index % n + 1,
        }:
            return False
    return seen == set(range(1, n + 1))


def _aggregate(value: object, n: int) -> bool:
    return bool(
        isinstance(value, dict)
        and value
        == {
            "first_constant": 0,
            "first_coefficients": [1] * n,
            "second_constant": n,
            "second_coefficients": [-1] * n,
        }
    )


def _square(value: object, n: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "lhs_coefficients",
        "square_coefficients",
    }:
        return False
    lhs = value["lhs_coefficients"]
    square = value["square_coefficients"]
    if lhs != [4, -4 * n, n * n] or square != [2, -n]:
        return False
    expanded = [square[0] ** 2, 2 * square[0] * square[1], square[1] ** 2]
    return expanded == lhs


def _equality(value: object, n: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "values",
        "term_norm_squared",
        "attained_constant_squared",
    }:
        return False
    try:
        values = [Fraction(item) for item in value["values"]]
        term = Fraction(value["term_norm_squared"])
        attained = Fraction(value["attained_constant_squared"])
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    if len(values) != n or values != [Fraction(1, 2)] * n:
        return False
    squares = [values[i] ** 2 + (1 - values[(i + 1) % n]) ** 2 for i in range(n)]
    return squares == [Fraction(1, 2)] * n and term == attained == Fraction(1, 2)


def _result(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "constant_squared",
        "dimension",
        "vectors",
        "aggregate",
        "completed_square",
        "equality_witness",
    }:
        return False
    n = value["dimension"]
    return bool(
        value["constant_squared"] == "1/2"
        and type(n) is int
        and 5 <= n <= 12
        and _vectors(value["vectors"], n)
        and _aggregate(value["aggregate"], n)
        and _square(value["completed_square"], n)
        and _equality(value["equality_witness"], n)
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
        text = path.read_text()
        markers = [
            line[12:].strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        ]
        bound = json.loads(markers[0]) if len(markers) == 1 else None
    except (OSError, UnicodeError, ValueError, RecursionError):
        return False
    folded = text.casefold()
    return bound == result and all(
        term in folded for term in ("cyclic", "triangle", "square", "sharp")
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
    correct = bool(
        protocol
        and isinstance(submission, dict)
        and _source_is_bound()
        and submission.get("scope") == SCOPE
        and submission.get("claimed_assurance") == "COMPUTED"
        and _result(submission.get("result"))
        and _evidence(submission.get("evidence"), submission.get("result"))
        and any(
            "not" in item.casefold() and "proof assistant" in item.casefold()
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
