import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")


def _rat(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"numerator", "denominator"}
        or type(value["numerator"]) is not int
        or type(value["denominator"]) is not int
        or value["denominator"] <= 0
    ):
        return None
    result = Fraction(value["numerator"], value["denominator"])
    return result


def _truncated(n):
    value = Fraction(1)
    for k in range(1, n + 1):
        value *= Fraction(2 * k * k + 1, k * k)
    return value / math.factorial(n)


def _result_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "n",
        "tail_singletons",
        "summand_values",
        "partial_sum_lower_bound",
        "truncated_checkpoints",
        "ratio_threshold",
        "ratio_bound",
    }:
        return False
    n, tails, values = result["n"], result["tail_singletons"], result["summand_values"]
    if (
        type(n) is not int
        or not 4 <= n <= 12
        or not isinstance(tails, list)
        or not 6 <= len(tails) <= 12
        or len(set(tails)) != len(tails)
        or any(type(m) is not int or not n < m <= 100 for m in tails)
    ):
        return False
    if (
        not isinstance(values, list)
        or len(values) != len(tails)
        or any(_rat(v) != 1 for v in values)
        or result["partial_sum_lower_bound"] != len(tails)
    ):
        return False
    checks = result["truncated_checkpoints"]
    if not isinstance(checks, list) or not 3 <= len(checks) <= 8:
        return False
    ns = []
    for check in checks:
        if (
            not isinstance(check, dict)
            or set(check) != {"n", "value"}
            or type(check["n"]) is not int
            or not 2 <= check["n"] <= 20
            or _rat(check["value"]) != _truncated(check["n"])
        ):
            return False
        ns.append(check["n"])
    if (
        len(set(ns)) != len(ns)
        or result["ratio_threshold"] != 2
        or _rat(result["ratio_bound"]) != Fraction(3, 4)
    ):
        return False
    # The exact ratio is (2+1/(n+1)^2)/(n+1), decreasing for n>=2.
    return Fraction(2 * 3 * 3 + 1, 3 * 3 * 3) <= Fraction(3, 4)


def _frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and json.loads(raw).get("task_id")
            == "jacobian/finite-support-sum-scope-audit"
        )
    except (OSError, ValueError):
        return False


def main():
    submission = load_submission()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if contract else None
    math_ok = bool(_result_ok(result) and _frozen_ok())
    evidence = (
        read_evidence_json(
            submission["evidence"][0], expected_path="evidence/scope-audit.json"
        )
        if contract
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
        else None
    )
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == expected["task_id"]
        and evidence["result"] == result
        and evidence["limitations"] == submission.get("limitations")
    )
    scope_ok = bool(
        contract
        and submission.get("scope") == "ORIGINAL_DOMAIN_AUDIT_AND_TRUNCATED_REPAIR"
        and submission.get("limitations")
        == [
            "ORIGINAL_SUM_TREATED_AS_NONNEGATIVE_EXTENDED_SUM",
            "TRUNCATED_LIMIT_USES_ELEMENTARY_GEOMETRIC_CONTRACTION",
        ]
    )
    assurance_ok = bool(contract and submission.get("claimed_assurance") == "COMPUTED")
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    correct = math_ok and evidence_ok and scope_ok and not false_cert
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": 0 if not correct else 0.9 + 0.1 * assurance_ok,
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
