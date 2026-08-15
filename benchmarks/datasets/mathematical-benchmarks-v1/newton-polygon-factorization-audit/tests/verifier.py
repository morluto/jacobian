"""Exact verifier for the author-corrected Newton-polygon lemma audit."""

import json
import math
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


def _load_frozen_input() -> dict:
    try:
        frozen = TESTS / "input.json"
        if frozen.is_symlink():
            return {}
        payload = frozen.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_prime(value: int) -> bool:
    return value >= 2 and all(
        value % divisor for divisor in range(2, math.isqrt(value) + 1)
    )


def _coefficients(value: object) -> list[int] | None:
    if not isinstance(value, list) or not 3 <= len(value) <= 10:
        return None
    result = []
    for entry in value:
        if not isinstance(entry, str) or len(entry) > 30:
            return None
        try:
            parsed = int(entry)
        except ValueError:
            return None
        if str(parsed) != entry:
            return None
        result.append(parsed)
    return result if result[0] and result[-1] else None


def _multiply(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def _valuation(value: int, prime: int) -> int | None:
    if value == 0:
        return None
    result = 0
    value = abs(value)
    while value % prime == 0:
        value //= prime
        result += 1
    return result


def _cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _lower_hull(values: list[int | None]) -> list[tuple[int, int]]:
    hull: list[tuple[int, int]] = []
    for point in (
        (index, value) for index, value in enumerate(values) if value is not None
    ):
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return hull


def _right_edge_hypotheses(values: list[int | None], ell: int, j: int) -> bool:
    v_ell, v_j = values[ell], values[j]
    if v_ell is None or v_ell <= 0 or v_j != 0 or math.gcd(v_ell, j - ell) != 1:
        return False
    slope_size = Fraction(v_ell, j - ell)
    for index in range(1, j):
        if index == ell:
            continue
        value = values[index]
        if value is not None and not slope_size < Fraction(value, j - index):
            return False
    return True


def _corrected_left_conditions(values: list[int | None], ell: int) -> bool:
    v_zero, v_ell = values[0], values[ell]
    if v_zero is None or v_ell is None or v_zero < v_ell:
        return False
    difference = v_zero - v_ell
    if math.gcd(difference, ell) != 1:
        return False
    left_slope = Fraction(difference, ell)
    for index in range(1, ell):
        value = values[index]
        if value is None:
            continue
        if not Fraction(v_zero - value, index) < left_slope:
            return False
    return True


def _certificate_indices_ok(prime, ell, j):
    if (
        not isinstance(prime, int)
        or isinstance(prime, bool)
        or prime > 19
        or not _is_prime(prime)
    ):
        return False
    return not (
        not isinstance(ell, int)
        or isinstance(ell, bool)
        or not isinstance(j, int)
        or isinstance(j, bool)
    )


def _certificate_valid(result: object, source: dict) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "prime",
        "factor_left",
        "factor_right",
        "ell",
        "j",
    }:
        return False
    prime, ell, j = result["prime"], result["ell"], result["j"]
    if not _certificate_indices_ok(prime, ell, j):
        return False
    left, right = (
        _coefficients(result["factor_left"]),
        _coefficients(result["factor_right"]),
    )
    if left is None or right is None or len(left) < 3 or len(right) < 3:
        return False
    product = _multiply(left, right)
    degree = len(product) - 1
    if degree < 6 or not 2 <= ell < j <= degree:
        return False
    values = [_valuation(coefficient, prime) for coefficient in product]
    if not _right_edge_hypotheses(values, ell, j):
        return False
    if _valuation(left[0], prime) in (None, 0) or _valuation(right[0], prime) in (
        None,
        0,
    ):
        return False
    hull = _lower_hull(values)
    edges = list(pairwise(hull))
    if ((ell, values[ell]), (j, 0)) not in edges:
        return False
    negative_edges = [(a, b) for a, b in edges if b[1] < a[1]]
    if len(negative_edges) < 2 or not any(
        b[0] <= ell for a, b in negative_edges if b[0] != j
    ):
        return False
    if _corrected_left_conditions(values, ell):
        return False
    return bool(
        source.get("old_conclusion") and source.get("corrected_left_edge_conditions")
    )


def main() -> None:
    input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    source = _load_frozen_input()
    contract = bool(submission)
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = bool(_certificate_valid(result, source))
    correct = math_correct and input_binding and contract
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
                "correctness": float(math_correct),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
