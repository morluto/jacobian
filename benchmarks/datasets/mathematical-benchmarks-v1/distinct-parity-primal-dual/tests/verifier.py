import json
import math
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = ["FINITE_INTEGER_OPTIMIZATION", "NO_PROOF_ASSISTANT_REPLAY"]


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def frontier():
    rows = []
    for n in range(1, 46, 2):
        budget = 2025 - n * n
        m = (math.isqrt(1 + 4 * budget) - 1) // 2
        while m * (m + 1) > budget:
            m -= 1
        rows.append(
            {
                "odd_count": n,
                "max_even_count": m,
                "minimum_sum": m * (m + 1) + n * n,
                "objective": 5 * m + 7 * n,
            }
        )
    return rows


def exact_value(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(exact_value(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                exact_value(value, target)
                for value, target in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def frontier_matches(submitted, expected):
    if not isinstance(submitted, list) or len(submitted) != len(expected):
        return False
    expected_by_odd = {row["odd_count"]: row for row in expected}
    seen = set()
    for row in submitted:
        if not isinstance(row, dict) or "odd_count" not in row:
            return False
        key = row["odd_count"]
        if key in seen or key not in expected_by_odd:
            return False
        seen.add(key)
        if not exact_value(row, expected_by_odd[key]):
            return False
    return True


def valid(r):
    if not isinstance(r, dict) or set(r) != {
        "even_numbers",
        "odd_numbers",
        "objective",
        "frontier",
    }:
        return False
    evens, odds = r["even_numbers"], r["odd_numbers"]
    if not isinstance(evens, list) or not isinstance(odds, list):
        return False
    if any(type(x) is not int or x <= 0 or x % 2 for x in evens) or any(
        type(x) is not int or x <= 0 or x % 2 != 1 for x in odds
    ):
        return False
    if (
        len(evens) != len(set(evens))
        or len(odds) != len(set(odds))
        or sum(evens) + sum(odds) != 2025
    ):
        return False
    objective = 5 * len(evens) + 7 * len(odds)
    rows = frontier()
    return (
        type(r["objective"]) is int
        and r["objective"] == objective == 384
        and frontier_matches(r["frontier"], rows)
        and max(row["objective"] for row in rows) == 384
    )


def main():
    expected = json.loads((T / "expected.json").read_text())
    s = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        s,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    ev = (
        read_evidence_json(
            s["evidence"][0], expected_path="evidence/distinct-parity-certificate.json"
        )
        if contract
        else None
    )
    input_bound = frozen()
    math_ok = bool(isinstance(s, dict) and valid(s.get("result")))
    evidence_ok = bool(
        contract
        and input_bound
        and ev
        and set(ev) == {"schema_version", "task_id", "result", "limitations"}
        and ev.get("schema_version") == "1"
        and ev.get("task_id") == expected["task_id"]
        and ev.get("result") == s.get("result")
        and ev.get("limitations") == LIMITATIONS
    )
    scope_ok = bool(
        contract
        and input_bound
        and s.get("scope") == "ALL_DISTINCT_POSITIVE_PARITY_LISTS_SUMMING_TO_2025"
        and s.get("completeness") == "COMPLETE"
        and s.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(contract and s.get("claimed_assurance") == "COMPUTED")
    false_cert = false_verified_claim(s, verification_record_bound=False)
    correct = bool(
        contract
        and input_bound
        and math_ok
        and evidence_ok
        and scope_ok
        and not false_cert
    )
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
