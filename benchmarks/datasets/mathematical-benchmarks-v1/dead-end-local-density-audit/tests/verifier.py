import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


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
        frozen.get("source", {}).get("revision")
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


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(
        isinstance(submission, dict)
        and _result_is_valid(data.get("result"), _load_input())
    )
    correct = math_correct
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
