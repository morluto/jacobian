import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")

_FRACTION_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")


def _source() -> dict[str, Any]:
    try:
        raw = (TESTS / "input.json").read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, str) or not _FRACTION_RE.fullmatch(value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _result_fraction(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError):
        return None


def _krawtchouk(order: int, distance: int, *, q: int = 3, n: int = 6) -> int:
    total = 0
    for h in range(order + 1):
        if h <= distance and order - h <= n - distance:
            total += (
                (-1) ** h
                * (q - 1) ** (order - h)
                * math.comb(distance, h)
                * math.comb(n - distance, order - h)
            )
    return total


def _construction(
    data: dict[str, Any], source: dict[str, Any]
) -> tuple[bool, Fraction, Fraction]:
    words = data.get("codewords")
    if (
        source.get("word_length") != 6
        or source.get("alphabet") != [0, 1, 2]
        or not isinstance(words, list)
        or len(words) != 18
    ):
        return False, Fraction(0), Fraction(0)
    # Validate element types before constructing the set so unhashable
    # codewords (arrays/objects) are rejected cleanly instead of raising.
    if not all(isinstance(word, str) for word in words):
        return False, Fraction(0), Fraction(0)
    if len(set(words)) != 18:
        return False, Fraction(0), Fraction(0)
    if any(len(word) != 6 or not set(word) <= set("012") for word in words):
        return False, Fraction(0), Fraction(0)
    ordered = {4: 0, 6: 0}
    for i, left in enumerate(words):
        for j, right in enumerate(words):
            if i == j:
                continue
            distance = sum(a != b for a, b in zip(left, right, strict=True))
            if distance not in ordered:
                return False, Fraction(0), Fraction(0)
            ordered[distance] += 1
    a4 = Fraction(ordered[4], len(words))
    a6 = Fraction(ordered[6], len(words))
    distribution = data.get("distance_distribution")
    valid = bool(
        isinstance(distribution, dict)
        and set(distribution) == {"A0", "A4", "A6"}
        and distribution["A0"] == "1"
        and _fraction(distribution["A4"]) == a4
        and _fraction(distribution["A6"]) == a6
        and 1 + a4 + a6 == len(words)
    )
    return valid, a4, a6


def _collect_krawtchouk_rows(
    rows: object,
) -> dict[int, dict[str, object]] | None:
    if not isinstance(rows, list) or len(rows) != 2:
        return None
    by_order: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"order", "K0", "K4", "K6"}:
            return None
        order = row["order"]
        if not isinstance(order, int) or isinstance(order, bool) or order in by_order:
            return None
        by_order[order] = row
    if set(by_order) != {1, 2}:
        return None
    return by_order


def _inequalities_valid(
    by_order: dict[int, dict[str, object]], a4: Fraction, a6: Fraction
) -> dict[int, tuple[Fraction, Fraction, Fraction]] | None:
    inequalities: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
    for order, row in by_order.items():
        expected = tuple(_krawtchouk(order, distance) for distance in (0, 4, 6))
        supplied = (row["K0"], row["K4"], row["K6"])
        if any(
            not isinstance(item, int) or isinstance(item, bool) for item in supplied
        ):
            return None
        if supplied != expected:
            return None
        coefficients = tuple(Fraction(item) for item in expected)
        inequalities[order] = coefficients
        if coefficients[0] + coefficients[1] * a4 + coefficients[2] * a6 < 0:
            return None
    return inequalities


def _upper_bound(value: object, a4: Fraction, a6: Fraction) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "krawtchouk_rows",
        "dual_multipliers",
        "bound",
    }:
        return False
    by_order = _collect_krawtchouk_rows(value["krawtchouk_rows"])
    if by_order is None:
        return False
    inequalities = _inequalities_valid(by_order, a4, a6)
    if inequalities is None:
        return False
    multipliers = value["dual_multipliers"]
    if not isinstance(multipliers, dict) or set(multipliers) != {"order_1", "order_2"}:
        return False
    m1 = _result_fraction(multipliers["order_1"])
    m2 = _result_fraction(multipliers["order_2"])
    if m1 is None or m2 is None or m1 < 0 or m2 < 0:
        return False
    combined = tuple(
        m1 * inequalities[1][index] + m2 * inequalities[2][index] for index in range(3)
    )
    # The dual certificate proves 18 - M = 17 - A4 - A6 >= 0.
    return bool(value["bound"] == 18 and combined == (17, -1, -1))


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    source = _source()
    result = data.get("result")
    result_data = result if isinstance(result, dict) else {}
    construction, a4, a6 = _construction(result_data, source)
    upper = bool(
        construction
        and _upper_bound(result_data.get("upper_bound_certificate"), a4, a6)
    )
    math_correct = bool(
        isinstance(submission, dict) and input_bound and construction and upper
    )
    correct = math_correct
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(correct),
            },
            sort_keys=True,
        )
        + "\n"
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
