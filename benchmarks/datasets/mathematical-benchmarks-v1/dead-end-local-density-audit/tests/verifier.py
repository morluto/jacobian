import json
import math
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
LIMITATION = (
    "The checker does not verify Euler-product convergence, the global "
    "asymptotic-density formula, or the upstream Lean development."
)


def _load_input() -> dict[str, Any]:
    try:
        visible = WORKSPACE / "input.json"
        hidden = TESTS / "input.json"
        if visible.is_symlink() or hidden.is_symlink():
            return {}
        raw = hidden.read_bytes()
        if visible.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % divisor for divisor in range(2, math.isqrt(n) + 1))


def _expected(case: dict[str, Any]) -> dict[str, Any] | None:
    case_id, p, b, digits = (
        case.get("case_id"),
        case.get("p"),
        case.get("b"),
        case.get("digits"),
    )
    if (
        not isinstance(case_id, str)
        or type(p) is not int
        or type(b) is not int
        or not _is_prime(p)
        or b < 2
        or not isinstance(digits, list)
        or not digits
        or any(type(d) is not int or d < 0 or d >= b for d in digits)
        or digits != sorted(set(digits))
    ):
        return None
    modulus = p * p
    forbidden = sorted(
        r
        for r in range(modulus)
        if r % modulus == 0 or any((b * r + digit) % modulus == 0 for digit in digits)
    )
    valid = modulus - len(forbidden)
    divisor = math.gcd(valid, modulus)
    branch = (
        "INVERTIBLE"
        if b % p
        else "SINGLY_DIVISIBLE"
        if b % modulus
        else "SQUARE_DIVISIBLE"
    )
    return {
        "case_id": case_id,
        "branch": branch,
        "forbidden_residues": forbidden,
        "valid_count": valid,
        "density_numerator": valid // divisor,
        "density_denominator": modulus // divisor,
    }


def _result_is_valid(result: object, frozen: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or set(result) != {"cases"}:
        return False
    source_cases = frozen.get("cases")
    submitted = result.get("cases")
    if (
        frozen.get("assurance_ceiling") != "COMPUTED"
        or frozen.get("source", {}).get("revision")
        != "80fc9124841a1f37a167d227d00780479d04f701"
        or not isinstance(source_cases, list)
        or not isinstance(submitted, list)
        or len(submitted) != len(source_cases)
    ):
        return False
    expected = [_expected(case) for case in source_cases if isinstance(case, dict)]
    if any(item is None for item in expected):
        return False
    if any(
        not isinstance(item, dict)
        or type(item.get("valid_count")) is not int
        or type(item.get("density_numerator")) is not int
        or type(item.get("density_denominator")) is not int
        or not isinstance(item.get("forbidden_residues"), list)
        or any(type(residue) is not int for residue in item["forbidden_residues"])
        for item in submitted
    ):
        return False
    by_id = {
        item.get("case_id"): item
        for item in submitted
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    if len(by_id) != len(submitted):
        return False
    return all(by_id.get(item["case_id"]) == item for item in expected if item)


def _evidence_is_valid(evidence: object) -> bool:
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        if target.stat().st_size > 1_048_576:
            return False
        text = target.read_text().strip()
    except (OSError, UnicodeError):
        return False
    return len(text) >= 20


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(
        contract and _result_is_valid(data.get("result"), _load_input())
    )
    evidence_valid = bool(math_correct and _evidence_is_valid(data.get("evidence")))
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(contract and LIMITATION in data.get("limitations", []))
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
