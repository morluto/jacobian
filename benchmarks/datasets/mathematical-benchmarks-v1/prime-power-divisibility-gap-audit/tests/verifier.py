from __future__ import annotations

import json
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    false_verified_claim,
    is_regular_bounded_file,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")
EVIDENCE_PATH = "evidence/divisibility-audit.json"
LIMITATION = "The countermodel refutes the frozen divisibility inference; it does not adjudicate the source theorem."
# Keep exponent-form integers within Python's default raw-integer parsing ceiling.
MAX_INTEGER_DIGITS = sys.int_info.default_max_str_digits


class _JsonFloat(float):
    """Preserve a JSON decimal token while remaining schema-compatible."""

    def __new__(cls, value: str) -> _JsonFloat:
        instance = super().__new__(cls, value)
        instance.lexeme = value
        return instance


def _bounded_json_float(value: str) -> int | _JsonFloat:
    try:
        exact = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"invalid JSON number: {value}") from error
    if not exact.is_finite() or (exact != 0 and exact.adjusted() >= MAX_INTEGER_DIGITS):
        raise ValueError(f"out-of-range JSON number: {value}")
    if exact == exact.to_integral_value():
        return int(exact)
    parsed = _JsonFloat(value)
    if not math.isfinite(parsed):
        raise ValueError(f"out-of-range JSON number: {value}")
    return parsed


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_untrusted_json(path: Path) -> object | None:
    try:
        return json.loads(
            path.read_text(),
            parse_float=_bounded_json_float,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, ValueError, RecursionError, MemoryError):
        return None


def _frozen() -> dict[str, Any]:
    try:
        value = json.loads((TESTS / "input.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _submission() -> dict[str, Any] | None:
    path = WORKSPACE / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    value = _read_untrusted_json(path)
    return value if isinstance(value, dict) else None


def _integer(
    value: object, *, minimum: int | None = None, maximum: int | None = None
) -> int | None:
    if type(value) is int:
        integer = value
    if isinstance(value, _JsonFloat):
        try:
            exact = Decimal(value.lexeme)
        except InvalidOperation:
            return None
        if (
            not exact.is_finite()
            or exact != exact.to_integral_value()
            or (exact != 0 and exact.adjusted() >= MAX_INTEGER_DIGITS)
            or (minimum is not None and exact < minimum)
            or (maximum is not None and exact > maximum)
        ):
            return None
        integer = int(exact)
    elif type(value) is not int:
        return None
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
        or type(maximum_modulus) is not int
        or maximum_modulus < 4
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
        and total % prime == 0
        and total % modulus != 0
        and valuation_modulus == exponent == reported_modulus_valuation
        and valuation_total == reported_total_valuation
        and 1 <= valuation_total < valuation_modulus
        and value["local_statement"] == "ALL_CYCLE_SUMS_DIVISIBLE_BY_P"
        and value["global_statement"] == "TOTAL_NOT_DIVISIBLE_BY_M"
        and value["missing_condition"] == "PRIME_POWER_MULTIPLICITIES"
    )


def _evidence(submission: dict[str, Any], *, expected_task_id: str) -> bool:
    evidence = submission.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    evidence_path = resolve_evidence(evidence[0], expected_path=EVIDENCE_PATH)
    if evidence_path is None:
        return False
    payload = _read_untrusted_json(evidence_path)
    return bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload["schema_version"] == "1"
        and payload["task_id"] == expected_task_id
        and payload["task_id"] == submission.get("task_id")
        and json.dumps(payload["result"], sort_keys=True, separators=(",", ":"))
        == json.dumps(submission.get("result"), sort_keys=True, separators=(",", ":"))
        and json.dumps(payload["limitations"], separators=(",", ":"))
        == json.dumps(submission.get("limitations"), separators=(",", ":"))
    )


def main() -> None:
    submission = _submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    expected = json.loads((TESTS / "expected.json").read_text())
    envelope_valid = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    limitations_correct = data.get("limitations") == [LIMITATION]
    contract = bool(envelope_valid and limitations_correct)
    math_correct = _result(data.get("result"), _frozen())
    evidence_valid = _evidence(data, expected_task_id=expected["task_id"])
    scope_correct = data.get("scope") == expected["required_scope"]
    assurance_correct = bool(
        data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        input_bound
        and contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "input_binding": float(input_bound),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "protocol_compliance": float(contract),
                "limitations_accuracy": float(limitations_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(out / "reward.json")


if __name__ == "__main__":
    main()
