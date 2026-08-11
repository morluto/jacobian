import json
from math import gcd
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")


def _json_equal(a, b):
    """Deep equality that rejects bool/int coercion (True == 1 in Python)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(
            _json_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    return a == b


def _frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and isinstance(json.loads(raw), dict)
        )
    except (OSError, ValueError):
        return False


def _prime(n):
    return (
        type(n) is int
        and 2 <= n <= 97
        and all(n % d for d in range(2, int(n**0.5) + 1))
    )


def _rows(value):
    if not isinstance(value, list) or not 3 <= len(value) <= 8:
        return None
    out = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"prime", "exponents"}:
            return None
        p, exps = row["prime"], row["exponents"]
        if (
            not _prime(p)
            or not isinstance(exps, list)
            or len(exps) != 4
            or any(type(e) is not int or not 0 <= e <= 6 for e in exps)
        ):
            return None
        out.append((p, exps))
    return out if [p for p, _ in out] == sorted({p for p, _ in out}) else None


def _numbers(rows):
    values = [1, 1, 1, 1]
    for p, exps in rows:
        for i, e in enumerate(exps):
            values[i] *= p**e
    return values


def _countermodel(rows):
    parsed = _rows(rows)
    if not parsed:
        return False
    values = _numbers(parsed)
    has_zero_min_with_positive_max = any(
        min(exps) == 0 and max(exps) > 0 for _, exps in parsed
    )
    has_positive_min = any(min(exps) > 0 for _, exps in parsed)
    return has_zero_min_with_positive_max and has_positive_min and gcd(*values) > 1


def _repair(rows):
    parsed = _rows(rows)
    if (
        not parsed
        or len({exps.index(0) for _, exps in parsed if exps.count(0) == 1}) < 2
    ):
        return False
    n = 1
    for p, exps in parsed:
        k = max(exps)
        if not 1 <= k <= 6 or min(exps) != 0 or sum(exps) != 3 * k:
            return False
        n *= p**k
    values = _numbers(parsed)
    lcm = 1
    for value in values:
        lcm = lcm * value // gcd(lcm, value)
    return (
        gcd(*values) == 1
        and lcm == n
        and values[0] * values[1] * values[2] * values[3] == n**3
    )


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
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = bool(
        isinstance(result, dict)
        and set(result) == {"defect", "countermodel", "repair"}
        and result["defect"] == "EXISTS_PRIME_MIN_ZERO_IS_TOO_WEAK"
        and _countermodel(result["countermodel"])
        and _repair(result["repair"])
        and _frozen_ok()
    )
    evidence = None
    if (
        contract
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
    ):
        evidence = read_evidence_json(
            submission["evidence"][0], expected_path="evidence/valuation-audit.json"
        )
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == expected["task_id"]
        and _json_equal(evidence["result"], result)
        and evidence["limitations"] == submission.get("limitations")
    )
    scope_ok = bool(
        contract
        and submission.get("scope")
        == "FROZEN_VALUATION_TRANSLATION_AND_SUBMITTED_TABLES"
        and submission.get("limitations")
        == [
            "SOURCE_FINAL_SUMMATION_NOT_ASSESSED",
            "INTEGER_RECONSTRUCTION_CHECKED_ONLY_FOR_SUBMITTED_PRIMES",
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
                "reward": float(correct),
                "false_certification": false_cert,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
