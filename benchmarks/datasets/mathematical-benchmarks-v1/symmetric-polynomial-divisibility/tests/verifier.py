import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
ZERO = (0, 0, 0, 0)
MAX_EVIDENCE_BYTES = 1_048_576


def _load_frozen_input():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        raw = frozen.read_bytes()
        if workspace.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _parse_polynomial(value, maximum_degree):
    if not isinstance(value, list) or len(value) > 70:
        return None
    result = {}
    order = []
    for term in value:
        if not isinstance(term, dict) or set(term) != {"exponents", "coefficient"}:
            return None
        exponents = term["exponents"]
        if (
            not isinstance(exponents, list)
            or len(exponents) != 4
            or any(type(item) is not int or item < 0 for item in exponents)
            or sum(exponents) > maximum_degree
        ):
            return None
        try:
            coefficient = Fraction(term["coefficient"])
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        if coefficient.denominator != 1:
            return None
        exponent = tuple(exponents)
        if (
            coefficient == 0
            or str(coefficient) != term["coefficient"]
            or exponent in result
        ):
            return None
        result[exponent] = coefficient
        order.append(exponent)
    return result if order == sorted(order) else None


def _add(left, right):
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _mul(left, right):
    result = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(x + y for x, y in zip(left_exp, right_exp, strict=True))
            result[exponent] = (
                result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def _variable(index, degree=1):
    exponent = [0, 0, 0, 0]
    exponent[index] = degree
    return {tuple(exponent): Fraction(1)}


def _generators_and_target():
    linear = {}
    quadratic = {}
    target = {}
    for index in range(4):
        linear = _add(linear, _variable(index))
        quadratic = _add(quadratic, _variable(index, 2))
        target = _add(target, _variable(index, 4))
    target[(1, 1, 1, 1)] = Fraction(4)
    return linear, quadratic, target


def _result_is_valid(result, frozen):
    if not isinstance(result, dict) or set(result) != {
        "variables",
        "generator_multipliers",
        "identity_conclusion",
        "divisibility_conclusion",
    }:
        return False
    if result["variables"] != frozen.get("variables") or result["variables"] != [
        "a",
        "b",
        "c",
        "d",
    ]:
        return False
    multipliers = result["generator_multipliers"]
    if not isinstance(multipliers, list) or len(multipliers) != 2:
        return False
    parsed = [
        _parse_polynomial(value, frozen.get("maximum_multiplier_degree"))
        for value in multipliers
    ]
    if any(value is None for value in parsed):
        return False
    linear, quadratic, target = _generators_and_target()
    reconstructed = _add(_mul(linear, parsed[0]), _mul(quadratic, parsed[1]))
    return bool(
        reconstructed == target
        and result["identity_conclusion"] == "TARGET_IN_HYPOTHESIS_IDEAL"
        and result["divisibility_conclusion"] == "TARGET_DIVISIBLE_BY_N"
    )


def _evidence_matches(evidence):
    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt")
    ):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        if target.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        text = target.read_text().casefold()
    except (OSError, UnicodeError):
        return False
    return (
        len(text) >= 120
        and all(word in text for word in ("identity", "coefficient", "divisible"))
        and re.search(r"(?:linear|quadratic).{0,120}(?:generator|multiple)", text)
        and re.search(r"(?:integral|integer).{0,120}(?:divis|modular)", text)
    )


def main():
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    frozen = _load_frozen_input()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _result_is_valid(data.get("result"), frozen))
    evidence_valid = bool(
        contract and math_correct and _evidence_matches(data.get("evidence"))
    )
    scope_text = (
        data.get("scope").casefold() if isinstance(data.get("scope"), str) else ""
    )
    scope_correct = bool(
        contract
        and all(term in scope_text for term in ("ideal", "integer", "divis"))
        and not re.search(
            r"(?:\b(?:no|not|without|excluding|exclude|only)\b"
            r"[\s_-]{0,20}(?:ideal|integer|divis)|"
            r"\bno[_-](?:ideal|integer|divis))",
            scope_text,
        )
    )
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitation_correct = bool(
        contract
        and any(
            isinstance(item, str)
            and "proof assistant" in item.casefold()
            and re.search(r"\b(?:not|no|doesn['']?t|without|cannot)\b", item, re.I)
            for item in data.get("limitations", [])
        )
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitation_correct
        and not false_certification
    )
    reward = 0.0 if not correct else 0.8 + 0.1 * scope_correct + 0.1 * assurance_correct
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
