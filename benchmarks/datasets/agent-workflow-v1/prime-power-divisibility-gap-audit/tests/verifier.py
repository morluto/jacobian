import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")
EVIDENCE_PATH = "evidence/divisibility-audit.json"
LIMITATION = "The countermodel refutes the frozen divisibility inference; it does not adjudicate the source theorem."


def _frozen() -> dict[str, Any]:
    try:
        visible, hidden = WORKSPACE / "input.json", TESTS / "input.json"
        if (
            visible.is_symlink()
            or hidden.is_symlink()
            or visible.read_bytes() != hidden.read_bytes()
        ):
            return {}
        value = json.loads(hidden.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _integer(value: object) -> int | None:
    return value if type(value) is int else None


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
    if not (
        _prime(prime)
        and 2 <= exponent <= 6
        and 1 <= factor <= 20
        and math.gcd(prime, factor) == 1
    ):
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
    prior_sum = 0
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"multiplicity", "cycle_sum"}:
            return False
        multiplicity = _integer(group["multiplicity"])
        cycle_sum = _integer(group["cycle_sum"])
        if (
            multiplicity is None
            or cycle_sum is None
            or multiplicity < 1
            or cycle_sum <= prior_sum
            or cycle_sum % prime
        ):
            return False
        prior_sum = cycle_sum
        multiplicity_sum += multiplicity
        recomputed_total += multiplicity * cycle_sum
    valuation_modulus = _valuation(modulus, prime)
    valuation_total = _valuation(total, prime)
    return bool(
        multiplicity_sum == cycle_count
        and recomputed_total == total
        and total % prime == 0
        and total % modulus != 0
        and valuation_modulus == exponent == value["p_valuation_modulus"]
        and valuation_total == value["p_valuation_total"]
        and 1 <= valuation_total < valuation_modulus
        and value["local_statement"] == "ALL_CYCLE_SUMS_DIVISIBLE_BY_P"
        and value["global_statement"] == "TOTAL_NOT_DIVISIBLE_BY_M"
        and value["missing_condition"] == "PRIME_POWER_MULTIPLICITIES"
    )


def _evidence(submission: dict[str, Any]) -> bool:
    evidence = submission.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    payload = read_evidence_json(evidence[0], expected_path=EVIDENCE_PATH)
    return bool(
        isinstance(payload, dict)
        and set(payload) == {"schema_version", "task_id", "result", "limitations"}
        and payload["schema_version"] == "1"
        and payload["task_id"] == submission.get("task_id")
        and payload["result"] == submission.get("result")
        and payload["limitations"] == submission.get("limitations")
    )


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
    math_correct = bool(contract and _result(data.get("result"), _frozen()))
    evidence_valid = bool(math_correct and _evidence(data))
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and LIMITATION in data.get("limitations", [])
        and not false_certification
    )
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
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
