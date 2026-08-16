from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = (Path("/app"), Path("/tests"))


def _frozen() -> dict[str, Any]:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _integer(
    value: object, *, minimum: int | None = None, maximum: int | None = None
) -> int | None:
    if type(value) is not int:
        return None
    integer = value
    if minimum is not None and integer < minimum:
        return None
    if maximum is not None and integer > maximum:
        return None
    return integer


def _prime(value: int) -> bool:
    return value >= 2 and all(
        value % divisor for divisor in range(2, math.isqrt(value) + 1)
    )


def _valuation(value: int, prime: int) -> int:
    exponent = 0
    while value % prime == 0:
        value //= prime
        exponent += 1
    return exponent


def _result(value: object, frozen: dict[str, Any]) -> bool:
    fields = {
        "prime",
        "exponent",
        "coprime_factor",
        "modulus",
        "cycle_count",
        "cycle_groups",
        "total_sum",
        "p_valuation_modulus",
        "p_valuation_total",
        "local_statement",
        "global_statement",
        "missing_condition",
    }
    maximum_modulus = frozen.get("maximum_modulus")
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or frozen.get("human_score") != 0
        or (type(maximum_modulus) is not int)
        or (maximum_modulus < 4)
    ):
        return False
    prime = _integer(value["prime"], minimum=2, maximum=29)
    exponent = _integer(value["exponent"], minimum=2, maximum=6)
    factor = _integer(value["coprime_factor"], minimum=1, maximum=1)
    modulus = _integer(value["modulus"], minimum=4, maximum=maximum_modulus)
    cycle_count = _integer(
        value["cycle_count"], minimum=4, maximum=maximum_modulus // 2
    )
    total = _integer(value["total_sum"], minimum=1)
    if None in {prime, exponent, factor, modulus, cycle_count, total}:
        return False
    if not (_prime(prime) and factor == 1):
        return False
    if modulus != prime**exponent * factor or cycle_count != modulus // prime:
        return False
    groups = value["cycle_groups"]
    if not isinstance(groups, list) or not 2 <= len(groups) <= 6:
        return False
    multiplicity_sum = 0
    recomputed_total = 0
    seen_sums: set[int] = set()
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"multiplicity", "cycle_sum"}:
            return False
        multiplicity = _integer(group["multiplicity"], minimum=1, maximum=cycle_count)
        cycle_sum = _integer(group["cycle_sum"], minimum=1)
        if (
            multiplicity is None
            or cycle_sum is None
            or cycle_sum in seen_sums
            or cycle_sum % prime
        ):
            return False
        seen_sums.add(cycle_sum)
        multiplicity_sum += multiplicity
        recomputed_total += multiplicity * cycle_sum
    reported_modulus_valuation = _integer(
        value["p_valuation_modulus"], minimum=2, maximum=exponent
    )
    reported_total_valuation = _integer(
        value["p_valuation_total"], minimum=1, maximum=exponent - 1
    )
    if None in {reported_modulus_valuation, reported_total_valuation}:
        return False
    valuation_modulus = _valuation(modulus, prime)
    valuation_total = _valuation(total, prime)
    return bool(
        multiplicity_sum == cycle_count
        and recomputed_total == total
        and (total % prime == 0)
        and (total % modulus != 0)
        and (valuation_modulus == exponent == reported_modulus_valuation)
        and (valuation_total == reported_total_valuation)
        and (1 <= valuation_total < valuation_modulus)
        and (value["local_statement"] == "ALL_CYCLE_SUMS_DIVISIBLE_BY_P")
        and (value["global_statement"] == "TOTAL_NOT_DIVISIBLE_BY_M")
        and (value["missing_condition"] == "PRIME_POWER_MULTIPLICITIES")
    )


def main() -> None:
    submission = load_submission(WORKSPACE / "submission.json")
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    contract = bool(submission)
    math_correct = _result(data.get("result"), _frozen())
    correct = bool(input_bound and contract and math_correct)
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
                "protocol_compliance": float(contract),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(out / "reward.json")


if __name__ == "__main__":
    main()
