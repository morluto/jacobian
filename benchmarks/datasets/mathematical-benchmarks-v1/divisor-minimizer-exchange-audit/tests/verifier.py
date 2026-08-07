import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

T = Path("/tests")
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


def partitions(total: int, cap: int | None = None):
    cap = total if cap is None else min(cap, total)
    if total == 0:
        yield ()
        return
    for first in range(cap, 0, -1):
        for tail in partitions(total - first, first):
            yield (first, *tail)


def candidate(partition: tuple[int, ...]) -> int:
    value = 1
    for prime, part in zip(PRIMES[: len(partition)], partition, strict=True):
        value *= prime ** ((1 << part) - 1)
    return value


def table(total: int) -> dict[tuple[int, ...], int]:
    return {part: candidate(part) for part in partitions(total)}


def factorization(value: int) -> list[dict[str, int]]:
    factors = []
    remaining = value
    for prime in PRIMES:
        exponent = 0
        while remaining % prime == 0:
            exponent += 1
            remaining //= prime
        if exponent:
            factors.append({"prime": prime, "exponent": exponent})
    return factors if remaining == 1 else []


def submitted_table(value: object) -> dict[tuple[int, ...], int] | None:
    if not isinstance(value, list):
        return None
    result = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {"partition", "value"}:
            return None
        part, val = item["partition"], item["value"]
        if (
            not isinstance(part, list)
            or any(type(x) is not int for x in part)
            or type(val) is not int
        ):
            return None
        key = tuple(part)
        if key in result:
            return None
        result[key] = val
    return result


def result_valid(result: object) -> bool:
    required = {
        "k",
        "current_minimizer",
        "current_factors",
        "current_divisor_count",
        "current_candidates",
        "next_minimizer",
        "next_factors",
        "next_divisor_count",
        "next_candidates",
        "quotient",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    current, following = table(12), table(13)
    current_min, next_min = min(current.values()), min(following.values())
    return bool(
        result["k"] == 12
        and submitted_table(result["current_candidates"]) == current
        and submitted_table(result["next_candidates"]) == following
        and result["current_minimizer"] == current_min
        and result["next_minimizer"] == next_min
        and result["current_factors"] == factorization(current_min)
        and result["next_factors"] == factorization(next_min)
        and result["current_divisor_count"] == 4096
        and result["next_divisor_count"] == 8192
        and next_min % current_min == 0
        and result["quotient"] == next_min // current_min
    )


def evidence_valid(value: object) -> bool:
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
        term in text
        for term in ("partition", "exponent", "prime", "minimal", "divides")
    )


def main() -> None:
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    math_ok = bool(contract and result_valid(submission.get("result")))
    evidence_ok = bool(contract and evidence_valid(submission.get("evidence")))
    scope_ok = bool(contract and submission.get("scope") == expected["required_scope"])
    assurance_ok = bool(contract and submission.get("claimed_assurance") == "COMPUTED")
    false = false_verified_claim(submission, verification_record_bound=False)
    correct = math_ok and not false
    reward = (
        0.0
        if not correct or not evidence_ok
        else 0.8 + 0.1 * scope_ok + 0.1 * assurance_ok
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
