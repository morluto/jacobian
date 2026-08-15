from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    witness_list_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


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
        first_variable = vector["first_variable"]
        second_constant = vector["second_constant"]
        second_variable = vector["second_variable"]
        if (
            type(index) is not int
            or type(first_variable) is not int
            or type(second_constant) is not int
            or type(second_variable) is not int
            or not 1 <= index <= n
            or index in seen
            or first_variable != index
            or second_constant != 1
            or second_variable != index % n + 1
        ):
            return False
        seen.add(index)
    return seen == set(range(1, n + 1))


def _aggregate(value: object, n: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "first_constant",
        "first_coefficients",
        "second_constant",
        "second_coefficients",
    }:
        return False
    first_constant = value["first_constant"]
    first_coefficients = value["first_coefficients"]
    second_constant = value["second_constant"]
    second_coefficients = value["second_coefficients"]
    return bool(
        type(first_constant) is int
        and first_constant == 0
        and isinstance(first_coefficients, list)
        and len(first_coefficients) == n
        and all(type(item) is int and item == 1 for item in first_coefficients)
        and type(second_constant) is int
        and second_constant == n
        and isinstance(second_coefficients, list)
        and len(second_coefficients) == n
        and all(type(item) is int and item == -1 for item in second_coefficients)
    )


def _square(value: object, n: int) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "lhs_coefficients",
        "square_coefficients",
    }:
        return False
    lhs = value["lhs_coefficients"]
    square = value["square_coefficients"]
    if not isinstance(lhs, list) or not isinstance(square, list):
        return False
    if any(type(item) is not int for item in lhs + square):
        return False
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
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
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
        or not witness_list_is_bound(value)
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


def _evaluate(submission: object) -> dict[str, float]:
    protocol_ok = isinstance(submission, dict)
    correct = bool(
        protocol_ok
        and _source_is_bound()
        and _result(submission.get("result"))
        and _evidence(submission.get("witness"), submission.get("result"))
    )
    reward = aggregate_reward(
        correctness=correct,
        witness_validity=correct,
        protocol_ok=protocol_ok,
    )
    return {
        "correctness": float(correct),
        "witness_validity": float(correct),
        "reward": reward,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
