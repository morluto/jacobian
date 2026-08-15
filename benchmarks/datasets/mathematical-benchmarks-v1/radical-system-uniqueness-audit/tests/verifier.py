import json
from pathlib import Path
from typing import Any

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
EXPECTED_ELIMINATION = [0, 0, -2, -1, 1]
EXPECTED_CLASSIFICATION = {
    -1: ("REJECTED", "PRINCIPAL_SIXTH_ROOT_REQUIRES_U_NONNEGATIVE"),
    0: ("REJECTED", "FOURTH_ROOT_RHS_REQUIRES_U_SQUARED_AT_LEAST_ONE"),
    2: ("ACCEPTED", "SATISFIES_ALL_ROOT_DOMAINS"),
}


def _load_bound_input() -> dict[str, Any]:
    try:
        visible = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if visible.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if visible.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _multiply(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def _factorization_is_complete(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    product = [1]
    for factor in value:
        if (
            not isinstance(factor, dict)
            or set(factor) != {"constant", "x_coefficient"}
            or type(factor["constant"]) is not int
            or type(factor["x_coefficient"]) is not int
            or factor["x_coefficient"] == 0
        ):
            return False
        product = _multiply(product, [factor["constant"], factor["x_coefficient"]])
    return product == EXPECTED_ELIMINATION


def _integer_roots(coefficients: list[int]) -> set[int]:
    # The submitted complete linear factorization bounds the rational roots;
    # this independent scan then reconstructs the distinct integer roots.
    roots = set()
    for value in range(-8, 9):
        if (
            sum(
                coefficient * value**power
                for power, coefficient in enumerate(coefficients)
            )
            == 0
        ):
            roots.add(value)
    return roots


def _classification_is_complete(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    observed: dict[int, tuple[object, object]] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"u", "status", "reason"}:
            return False
        u = item["u"]
        if type(u) is not int or u in observed:
            return False
        observed[u] = (item["status"], item["reason"])
    return observed == EXPECTED_CLASSIFICATION


def _solution_is_exact(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "a",
        "b",
        "c",
        "u",
        "equation_values",
    }:
        return False
    a, b, c, u = (value[name] for name in ("a", "b", "c", "u"))
    if not all(type(item) is int for item in (a, b, c, u)) or c == 0:
        return False
    if u < 0 or u**6 != b or u**2 - 1 < 0:
        return False
    # Avoid floating roots: the three source equations are equivalent to
    # these exact integer identities under the certified root domains.
    equations_hold = (
        a == c * (b - u**3 + c)
        and a == c * (u**3 + 1) ** 2
        and a == c * (u**2 - 1) ** 4
    )
    expected_values = [
        {"left": a // c, "right": b - u**3 + c},
        {"left": u**3 + 1, "right": u**3 + 1},
        {"left": u**2 - 1, "right": u**2 - 1},
    ]
    submitted_values = value["equation_values"]
    if not isinstance(submitted_values, list) or len(submitted_values) != 3:
        return False
    submitted_pairs = [
        (item["left"], item["right"])
        for item in submitted_values
        if isinstance(item, dict)
        and set(item) == {"left", "right"}
        and type(item["left"]) is int
        and type(item["right"]) is int
    ]
    expected_pairs = [(item["left"], item["right"]) for item in expected_values]
    return bool(
        equations_hold
        and len(submitted_pairs) == len(expected_pairs)
        and sorted(submitted_pairs) == sorted(expected_pairs)
    )


def _certificate_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "substitution",
        "elimination_coefficients",
        "linear_factors",
        "root_classification",
        "solutions",
    }:
        return False
    coefficients = value["elimination_coefficients"]
    if (
        not isinstance(coefficients, list)
        or not all(type(coefficient) is int for coefficient in coefficients)
        or value["substitution"] != "u=b^(1/6)"
        or coefficients != EXPECTED_ELIMINATION
        or not _factorization_is_complete(value["linear_factors"])
        or _integer_roots(coefficients) != set(EXPECTED_CLASSIFICATION)
        or not _classification_is_complete(value["root_classification"])
    ):
        return False
    solutions = value["solutions"]
    return bool(
        isinstance(solutions, list)
        and len(solutions) == 1
        and _solution_is_exact(solutions[0])
        and solutions[0]["u"] == 2
    )


def main() -> None:
    submission = load_submission()
    source = _load_bound_input()
    protocol_ok = submission is not None
    source_ok = bool(
        source.get("substitution") == "u = b^(1/6)"
        and source.get("source", {}).get("row_sha256")
        == "sha256:b9a10fbc445876cc412565550ef03722124d698b9ad52d0f1f6aacd0e97b823c"
    )
    math_correct = bool(
        protocol_ok and source_ok and _certificate_is_valid(submission.get("result"))
    )
    reward = aggregate_reward(
        correctness=math_correct,
        witness_validity=True,
        protocol_ok=protocol_ok,
    )
    output = Path("/logs/verifier/reward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(output)


if __name__ == "__main__":
    main()
