import json
import math
import re
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")


def _load_frozen_input():
    try:
        workspace, frozen = W / "input.json", E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        raw = frozen.read_bytes()
        if workspace.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _trim(poly):
    result = list(poly)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _integer(value):
    if type(value) is int:
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _integer_list(value):
    if not isinstance(value, list) or not value:
        return None
    normalized = [_integer(item) for item in value]
    return normalized if all(item is not None for item in normalized) else None


def _mul(left, right):
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return _trim(result)


def _divide_exact(dividend, divisor):
    remainder = _trim(dividend)
    divisor = _trim(divisor)
    if not divisor or divisor[-1] not in {1, -1} or len(remainder) < len(divisor):
        return None
    quotient = [0] * (len(remainder) - len(divisor) + 1)
    while len(remainder) >= len(divisor) and remainder != [0]:
        offset = len(remainder) - len(divisor)
        lead = remainder[-1] // divisor[-1]
        if lead * divisor[-1] != remainder[-1]:
            return None
        quotient[offset] += lead
        for index, coefficient in enumerate(divisor):
            remainder[index + offset] -= lead * coefficient
        remainder = _trim(remainder)
    return _trim(quotient) if remainder == [0] else None


def _cyclotomic(order, cache):
    if order in cache:
        return cache[order]
    polynomial = [-1] + [0] * (order - 1) + [1]
    for divisor in range(1, order):
        if order % divisor == 0:
            polynomial = _divide_exact(polynomial, _cyclotomic(divisor, cache))
            if polynomial is None:
                return None
    cache[order] = polynomial
    return polynomial


def _result_is_valid(result, frozen):
    required = {
        "leading_coefficient",
        "factors",
        "expanded_coefficients",
        "reciprocal_coefficients",
        "p_at_one",
        "reciprocal_scalar",
        "root_orbit_conclusion",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    leading = _integer(result["leading_coefficient"])
    p_at_one = _integer(result["p_at_one"])
    reciprocal_scalar = _integer(result["reciprocal_scalar"])
    if leading is None or leading == 0:
        return False
    factors = result["factors"]
    if not isinstance(factors, list) or not factors or len(factors) > 8:
        return False
    parsed, total = [], 0
    for factor in factors:
        if not isinstance(factor, dict) or set(factor) != {"order", "multiplicity"}:
            return False
        order, multiplicity = factor["order"], factor["multiplicity"]
        order = _integer(order)
        multiplicity = _integer(multiplicity)
        if (
            order is None
            or multiplicity is None
            or not 1 < order <= frozen.get("maximum_cyclotomic_order", 0)
            or not 1 <= multiplicity <= 8
        ):
            return False
        parsed.append((order, multiplicity))
        total += multiplicity
    if total > frozen.get("maximum_total_multiplicity", 0):
        return False
    cache = {1: [-1, 1]}
    product = [leading]
    for order, multiplicity in sorted(parsed):
        factor = _cyclotomic(order, cache)
        if factor is None or factor != list(reversed(factor)):
            return False
        for _ in range(multiplicity):
            product = _mul(product, factor)
    expanded = _integer_list(result["expanded_coefficients"])
    reciprocal = _integer_list(result["reciprocal_coefficients"])
    coefficients = frozen.get("coefficients")
    return bool(
        product == coefficients
        and expanded == product
        and reciprocal == list(reversed(product))
        and product == list(reversed(product))
        and p_at_one == sum(product)
        and p_at_one != 0
        and reciprocal_scalar == 1
        and result["root_orbit_conclusion"]
        == "INVERSION_CLOSED_WITH_EQUAL_MULTIPLICITIES"
    )


def _evidence_matches(evidence):
    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt")
    ):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    try:
        text = target.read_text().casefold() if target else ""
    except (OSError, UnicodeError):
        return False
    return bool(
        len(text) >= 180
        and "cyclotomic" in text
        and "inversion" in text
        and "root" in text
        and "orbit" in text
        and ("phi_1" in text or "phi1" in text)
        and "p(1)" in text
        and re.search(r"x\s*[-\u2212]\s*1", text)
        and ("coefficient symmetry" in text or "reciprocal" in text)
    )


def main():
    submission, frozen = load_submission(), _load_frozen_input()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _result_is_valid(submission.get("result"), frozen))
    evidence_valid = bool(
        contract and math_correct and _evidence_matches(submission.get("evidence"))
    )
    scope = submission.get("scope") if isinstance(submission, dict) else None
    scope_text = scope.casefold() if isinstance(scope, str) else ""
    scope_correct = bool(
        contract
        and ("degree 16" in scope_text or "degree_16" in scope_text)
        and "frozen" in scope_text
        and "cyclotomic" in scope_text
        and "general" in scope_text
        and "orbit" in scope_text
        and not re.search(
            r"\b(?:not|without|excluding|only)\b[^.]{0,50}\bgeneral\b", scope_text
        )
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = (
        submission.get("limitations", []) if isinstance(submission, dict) else []
    )
    limitation_correct = False
    if contract and isinstance(limitations, list):
        combined = " ".join(
            item.casefold() for item in limitations if isinstance(item, str)
        )
        negative_pattern = (
            r"\b(?:not|no|without|does not|doesn't|lacks?|"
            r"no\s+[^.]{0,20})\b[^.]{0,60}"
            r"\b(?:machine|formal(?:ly)?|proof[- ]assistant)\b"
        )
        limitation_correct = bool(
            re.search(r"\bunrestricted\b", combined)
            and re.search(negative_pattern, combined)
            and not re.search(
                r"\b(?:machine|formal(?:ly)?|proof[- ]assistant)\b[^.]{0,60}\b(?:verified|checked|proof)\b",
                re.sub(negative_pattern, "", combined),
            )
        )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        contract and math_correct and limitation_correct and not false_certification
    )
    reward = (
        0.0
        if not correct or not evidence_valid
        else 0.8 + 0.1 * scope_correct + 0.1 * assurance_correct
    )
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


if __name__ == "__main__":
    main()
