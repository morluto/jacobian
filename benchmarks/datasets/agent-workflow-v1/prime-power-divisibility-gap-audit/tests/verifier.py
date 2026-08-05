from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    false_verified_claim,
    is_regular_bounded_file,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")
EVIDENCE_PATH = "evidence/divisibility-audit.json"
LIMITATION = "The countermodel refutes the frozen divisibility inference; it does not adjudicate the source theorem."


class _JsonFloat(float):
    """Preserve a JSON decimal token while remaining schema-compatible."""

    def __new__(cls, value: str) -> _JsonFloat:
        instance = super().__new__(cls, value)
        instance.lexeme = value
        return instance


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
    try:
        value = json.loads(path.read_text(), parse_float=_JsonFloat)
    except (OSError, ValueError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _integer(value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, _JsonFloat):
        try:
            exact = Decimal(value.lexeme)
        except InvalidOperation:
            return None
        if exact.is_finite() and exact == exact.to_integral_value():
            return int(exact)
    return None


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
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or frozen.get("human_score") != 0
    ):
        return False
    prime = _integer(value["prime"])
    exponent = _integer(value["exponent"])
    factor = _integer(value["coprime_factor"])
    modulus = _integer(value["modulus"])
    cycle_count = _integer(value["cycle_count"])
    total = _integer(value["total_sum"])
    if None in {prime, exponent, factor, modulus, cycle_count, total}:
        return False
    if not (2 <= prime <= 29 and _prime(prime) and 2 <= exponent <= 6 and factor == 1):
        return False
    if (
        modulus != prime**exponent * factor
        or modulus > frozen.get("maximum_modulus")
        or cycle_count != modulus // prime
        or cycle_count < 4
    ):
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
        multiplicity = _integer(group["multiplicity"])
        cycle_sum = _integer(group["cycle_sum"])
        if (
            multiplicity is None
            or cycle_sum is None
            or multiplicity < 1
            or cycle_sum < 1
            or cycle_sum in seen_sums
            or cycle_sum % prime
        ):
            return False
        seen_sums.add(cycle_sum)
        multiplicity_sum += multiplicity
        recomputed_total += multiplicity * cycle_sum
    reported_modulus_valuation = _integer(value["p_valuation_modulus"])
    reported_total_valuation = _integer(value["p_valuation_total"])
    if total <= 0 or None in {reported_modulus_valuation, reported_total_valuation}:
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
    payload = read_evidence_json(evidence[0], expected_path=EVIDENCE_PATH)
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


if __name__ == "__main__":
    main()
