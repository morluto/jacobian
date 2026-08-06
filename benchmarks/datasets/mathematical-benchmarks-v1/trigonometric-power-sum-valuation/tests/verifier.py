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


def _valuation(value, prime):
    if type(value) is not int or value == 0:
        return None
    exponent = 0
    remaining = abs(value)
    while remaining % prime == 0:
        exponent += 1
        remaining //= prime
    return exponent


def _integer(value):
    if type(value) is int:
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _expected_terms(limit):
    values = [3, 7, 21]
    for n in range(3, limit + 1):
        values.append(7 * values[n - 1] - 14 * values[n - 2] + 7 * values[n - 3])
    return values


def _terms_are_valid(terms, values):
    if not isinstance(terms, list) or len(terms) != len(values):
        return False
    for n, (term, value) in enumerate(zip(terms, values, strict=True)):
        if not isinstance(term, dict) or set(term) != {
            "n",
            "value",
            "seven_adic_valuation",
            "required_valuation",
        }:
            return False
        normalized = {
            field: _integer(term.get(field))
            for field in ("n", "value", "seven_adic_valuation", "required_valuation")
        }
        if any(value is None for value in normalized.values()):
            return False
        if normalized != {
            "n": n,
            "value": value,
            "seven_adic_valuation": _valuation(value, 7),
            "required_valuation": n // 3,
        }:
            return False
    return True


def _induction_is_valid(cases):
    expected = [
        {"residue": 0, "coefficient_adjusted_offsets": [0, 0, 0]},
        {"residue": 1, "coefficient_adjusted_offsets": [1, 0, 0]},
        {"residue": 2, "coefficient_adjusted_offsets": [1, 1, 0]},
    ]
    if not isinstance(cases, list) or len(cases) != len(expected):
        return False
    normalized = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(
            case.get("coefficient_adjusted_offsets"), list
        ):
            return False
        residue = _integer(case.get("residue"))
        offsets = [_integer(value) for value in case["coefficient_adjusted_offsets"]]
        if residue is None or any(value is None for value in offsets):
            return False
        normalized.append({"residue": residue, "coefficient_adjusted_offsets": offsets})
    return sorted(normalized, key=lambda case: case["residue"]) == expected


def _result_is_valid(result, frozen):
    if not isinstance(result, dict) or set(result) != {
        "minimal_polynomial_descending",
        "initial_power_sums",
        "recurrence_coefficients",
        "terms",
        "induction_cases",
        "conclusion",
    }:
        return False
    limit = frozen.get("term_limit")
    if type(limit) is not int or limit != 24:
        return False
    for field in (
        "minimal_polynomial_descending",
        "initial_power_sums",
        "recurrence_coefficients",
    ):
        values = result[field]
        if not isinstance(values, list):
            return False
        normalized = [_integer(value) for value in values]
        if any(value is None for value in normalized):
            return False
        result[field] = normalized
    values = _expected_terms(limit)
    return bool(
        result["minimal_polynomial_descending"] == [1, -7, 14, -7]
        and result["initial_power_sums"] == [3, 7, 21]
        and result["recurrence_coefficients"] == [7, -14, 7]
        and _terms_are_valid(result["terms"], values)
        and _induction_is_valid(result["induction_cases"])
        and isinstance(result["conclusion"], str)
        and re.search(
            r"(?:\b|_)divis(?:ible|ibility)(?:\b|_)", result["conclusion"], re.I
        )
        and re.search(r"(?:\b|_)(?:positive|all)(?:\b|_)", result["conclusion"], re.I)
        and not re.search(
            r"\b(?:not|without|cannot|unknown|insufficient|fail(?:s|ure)?)\b",
            result["conclusion"],
            re.I,
        )
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
    return bool(
        len(text) >= 180
        and all(term in text for term in ("cubic", "7-adic", "valuation", "divis"))
        and ("recurrence" in text or "newton" in text or "power sum" in text)
        and ("induction" in text or "residue" in text or "valuation argument" in text)
        and "minimal polynomial" in text
    )


def _limitation_is_valid(limitations):
    if not isinstance(limitations, list):
        return False
    return any(
        isinstance(item, str)
        and "trigonometric" in item.casefold()
        and re.search(
            r"\b(?:not|doesn['']?t|cannot)\b.{0,100}\b(?:independently )?(?:verify|check|prove)\b",
            item,
            re.I,
        )
        for item in limitations
    )


def main():
    submission = load_submission()
    frozen = _load_frozen_input()
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
        contract
        and isinstance(submission.get("scope"), str)
        and all(
            term in submission["scope"].casefold()
            for term in ("cubic", "recurrence", "7-adic")
        )
        and not re.search(
            r"\b(?:not|doesn['']?t|cannot|without|except|excluding)\b",
            submission["scope"],
            re.I,
        )
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitation_correct = bool(
        contract and _limitation_is_valid(submission.get("limitations"))
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
