import json
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
    if (
        not isinstance(result, dict)
        or set(result) != required
        or type(result["leading_coefficient"]) is not int
        or result["leading_coefficient"] == 0
    ):
        return False
    factors = result["factors"]
    if not isinstance(factors, list) or not factors or len(factors) > 8:
        return False
    parsed, previous, total = [], 1, 0
    for factor in factors:
        if not isinstance(factor, dict) or set(factor) != {"order", "multiplicity"}:
            return False
        order, multiplicity = factor["order"], factor["multiplicity"]
        if (
            type(order) is not int
            or type(multiplicity) is not int
            or not previous < order <= frozen.get("maximum_cyclotomic_order", 0)
            or not 1 <= multiplicity <= 8
        ):
            return False
        parsed.append((order, multiplicity))
        previous, total = order, total + multiplicity
    if total > frozen.get("maximum_total_multiplicity", 0):
        return False
    cache = {1: [-1, 1]}
    product = [result["leading_coefficient"]]
    for order, multiplicity in parsed:
        factor = _cyclotomic(order, cache)
        if factor is None or factor != list(reversed(factor)):
            return False
        for _ in range(multiplicity):
            product = _mul(product, factor)
    coefficients = frozen.get("coefficients")
    return bool(
        product == coefficients
        and result["expanded_coefficients"] == product
        and result["reciprocal_coefficients"] == list(reversed(product))
        and product == list(reversed(product))
        and result["p_at_one"] == sum(product)
        and result["p_at_one"] != 0
        and result["reciprocal_scalar"] == 1
        and result["root_orbit_conclusion"]
        == "INVERSION_CLOSED_WITH_EQUAL_MULTIPLICITIES"
    )


def _evidence_matches(evidence):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    try:
        text = target.read_text().casefold() if target else ""
    except (OSError, UnicodeError):
        return False
    return all(word in text for word in ("cyclotomic", "inversion", "p(1)"))


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
    scope_correct = bool(
        contract and submission.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitation_correct = bool(
        contract
        and any(
            "unrestricted" in item.casefold() and "not" in item.casefold()
            for item in submission.get("limitations", [])
        )
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        contract and math_correct and limitation_correct and not false_certification
    )
    reward = (
        0
        if not correct
        else 0.7 + 0.1 * evidence_valid + 0.1 * scope_correct + 0.1 * assurance_correct
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
