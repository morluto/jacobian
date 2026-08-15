import json
import math
import re
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")


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


def main():
    submission = load_submission()
    frozen = _load_frozen_input()
    protocol_ok = submission is not None
    math_correct = bool(
        protocol_ok and _result_is_valid(submission.get("result"), frozen)
    )
    reward = aggregate_reward(
        correctness=math_correct,
        witness_validity=True,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
