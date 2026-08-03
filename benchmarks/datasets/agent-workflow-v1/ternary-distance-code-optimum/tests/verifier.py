import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


def _source() -> dict[str, Any]:
    try:
        raw = (TESTS / "input.json").read_bytes()
        if (WORKSPACE / "input.json").read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _fraction(value: object) -> Fraction | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if str(parsed) != value:
        return None
    return parsed


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
        or len(set(words)) != 18
        or any(
            not isinstance(word, str) or len(word) != 6 or set(word) > set("012")
            for word in words
        )
    ):
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


def _upper_bound(value: object, a4: Fraction, a6: Fraction) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "krawtchouk_rows",
        "dual_multipliers",
        "bound",
    }:
        return False
    rows = value["krawtchouk_rows"]
    if not isinstance(rows, list) or len(rows) != 2:
        return False
    by_order: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"order", "K0", "K4", "K6"}:
            return False
        order = row["order"]
        if not isinstance(order, int) or isinstance(order, bool) or order in by_order:
            return False
        by_order[order] = row
    if set(by_order) != {1, 2}:
        return False
    inequalities: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
    for order, row in by_order.items():
        expected = tuple(_krawtchouk(order, distance) for distance in (0, 4, 6))
        supplied = (row["K0"], row["K4"], row["K6"])
        if any(
            not isinstance(item, int) or isinstance(item, bool) for item in supplied
        ):
            return False
        if supplied != expected:
            return False
        coefficients = tuple(Fraction(item) for item in expected)
        inequalities[order] = coefficients
        if coefficients[0] + coefficients[1] * a4 + coefficients[2] * a6 < 0:
            return False
    multipliers = value["dual_multipliers"]
    if not isinstance(multipliers, dict) or set(multipliers) != {"order_1", "order_2"}:
        return False
    m1 = _fraction(multipliers["order_1"])
    m2 = _fraction(multipliers["order_2"])
    if m1 is None or m2 is None or m1 < 0 or m2 < 0:
        return False
    combined = tuple(
        m1 * inequalities[1][index] + m2 * inequalities[2][index] for index in range(3)
    )
    # The dual certificate proves 18 - M = 17 - A4 - A6 >= 0.
    return bool(value["bound"] == 18 and combined == (17, -1, -1))


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
    return len(text) >= 120 and all(
        term in text for term in ("18", "a4", "a6", "delsarte", "computed")
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion="OPTIMUM_CERTIFIED",
        verification_record="forbidden",
    )
    source = _source()
    result = data.get("result")
    result_data = result if isinstance(result, dict) else {}
    construction, a4, a6 = _construction(result_data, source)
    upper = bool(
        construction
        and _upper_bound(result_data.get("upper_bound_certificate"), a4, a6)
    )
    correct = bool(contract and construction and upper)
    evidence = bool(correct and _evidence(data.get("evidence")))
    scope = bool(
        contract
        and data.get("scope")
        == "ternary words of length 6 with distinct-pair agreements in {0,2}"
        and data.get("completeness") == "COMPLETE"
    )
    assurance = bool(contract and data.get("claimed_assurance") == "COMPUTED")
    limitations = bool(
        contract
        and isinstance(data.get("limitations"), list)
        and any(
            isinstance(item, str) and "formal proof" in item.casefold()
            for item in data["limitations"]
        )
    )
    false_verified = false_verified_claim(submission, verification_record_bound=False)
    passed = bool(
        correct
        and evidence
        and scope
        and assurance
        and limitations
        and not false_verified
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correct),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": float(passed),
                "false_certification": false_verified,
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
