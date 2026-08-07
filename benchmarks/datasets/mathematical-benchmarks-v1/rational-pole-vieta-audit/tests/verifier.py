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

TESTS = Path("/tests")


def _mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _add(a: list[int], b: list[int]) -> list[int]:
    out = [0] * max(len(a), len(b))
    for i, x in enumerate(a):
        out[i] += x
    for i, x in enumerate(b):
        out[i] += x
    return out


def _canonical_fraction(value: object) -> Fraction | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if str(parsed) == value else None


def _evidence_valid(value: object) -> bool:
    if not evidence_list_is_bound(value):
        return False
    target = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text().lower()
    except (OSError, UnicodeError):
        return False
    return len(text.split()) >= 55 and all(
        term in text for term in ("pole", "denominator", "nonzero", "vieta")
    )


def _result_valid(result: object) -> bool:
    required = {
        "denominator_coefficients",
        "combined_numerator_coefficients",
        "cleared_polynomial_coefficients",
        "pole_square_residuals",
        "root_sum",
        "diagnosis",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    denominator = [1]
    for k in range(1, 5):
        denominator = _mul(denominator, [-k, 0, 1])
    numerator = [0]
    residuals = []
    for k in range(1, 5):
        quotient = [1]
        residual = k
        for j in range(1, 5):
            if j != k:
                quotient = _mul(quotient, [-j, 0, 1])
                residual *= k - j
        numerator = _add(numerator, [k * value for value in quotient])
        residuals.append({"k": k, "residual": residual})
    cleared = _add(numerator, _mul([4, -2010], denominator))
    root_sum = -Fraction(cleared[-2], cleared[-1])
    return bool(
        result["denominator_coefficients"] == denominator
        and result["combined_numerator_coefficients"] == numerator
        and result["cleared_polynomial_coefficients"] == cleared
        and result["pole_square_residuals"] == residuals
        and all(item["residual"] != 0 for item in residuals)
        and _canonical_fraction(result["root_sum"]) == root_sum
        and result["diagnosis"]
        == "POLES_ARE_PLUS_MINUS_SQUARE_ROOTS_NOT_DENOMINATOR_PARAMETERS"
    )


def main() -> None:
    submission = load_submission()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    mathematical = _result_valid(result)
    evidence = bool(contract and _evidence_valid(submission.get("evidence")))
    scope = bool(contract and submission.get("scope") == expected["required_scope"])
    assurance = bool(contract and submission.get("claimed_assurance") == "COMPUTED")
    false = false_verified_claim(submission, verification_record_bound=False)
    correct = bool(contract and mathematical and not false)
    reward = 0.0 if not correct or not evidence else 0.8 + 0.1 * scope + 0.1 * assurance
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(mathematical),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
