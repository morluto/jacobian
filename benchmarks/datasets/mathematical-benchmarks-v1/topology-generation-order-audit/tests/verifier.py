import json
from itertools import combinations
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

TESTS = Path("/tests")


def _is_topology(value: object, n: int) -> bool:
    if (
        not isinstance(value, list)
        or any(type(item) is not int for item in value)
        or value != sorted(set(value))
    ):
        return False
    opens = set(value)
    full = (1 << n) - 1
    if (
        0 not in opens
        or full not in opens
        or any(item < 0 or item > full for item in opens)
    ):
        return False
    return all((a | b) in opens and (a & b) in opens for a in opens for b in opens)


def _all_topologies(n: int):
    middle = list(range(1, (1 << n) - 1))
    for size in range(len(middle) + 1):
        for chosen in combinations(middle, size):
            family = [0, *chosen, (1 << n) - 1]
            if _is_topology(family, n):
                yield set(family)


def _generated(inputs: list[list[int]], n: int) -> list[int] | None:
    required = set().union(*(set(item) for item in inputs))
    containing = [topology for topology in _all_topologies(n) if required <= topology]
    if not containing:
        return None
    return sorted(set.intersection(*containing))


def _evidence_valid(value: object) -> bool:
    if not evidence_list_is_bound(value):
        return False
    target = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text().lower()
    except (OSError, UnicodeError):
        return False
    return len(text.split()) >= 45 and all(
        term in text for term in ("topology", "contained", "containing", "open set")
    )


def _result_valid(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "universe_size",
        "input_topologies",
        "generated_topology",
        "common_subtopology",
        "witness_open_set",
        "diagnosis",
    }:
        return False
    n = result["universe_size"]
    inputs = result["input_topologies"]
    generated = result["generated_topology"]
    common = result["common_subtopology"]
    witness = result["witness_open_set"]
    if type(n) is not int or n not in {3, 4} or not isinstance(inputs, list):
        return False
    if len(inputs) < 2 or any(
        not _is_topology(item, n) or len(item) < 4 for item in inputs
    ):
        return False
    families = [set(item) for item in inputs]
    if len({tuple(item) for item in inputs}) != len(inputs):
        return False
    if not all(
        not (left <= right) and not (right <= left)
        for left, right in combinations(families, 2)
    ):
        return False
    expected_generated = _generated(inputs, n)
    if generated != expected_generated or not _is_topology(common, n):
        return False
    common_set = set(common)
    generated_set = set(generated)
    return bool(
        all(common_set <= family for family in families)
        and common_set != generated_set
        and type(witness) is int
        and witness in generated_set
        and witness not in common_set
        and result["diagnosis"]
        == "COMMON_SUBTOPOLOGY_DOES_NOT_ESTABLISH_LEAST_CONTAINING_TOPOLOGY"
    )


def main() -> None:
    submission = load_submission()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    mathematical = _result_valid(result)
    evidence = bool(contract and _evidence_valid(submission.get("evidence")))
    scope = bool(contract and submission.get("scope") == expected["required_scope"])
    assurance = bool(contract and submission.get("claimed_assurance") == "COMPUTED")
    false = false_verified_claim(submission, verification_record_bound=False)
    correct = bool(contract and mathematical and not false)
    reward = (
        0.0 if not correct else 0.7 + 0.1 * evidence + 0.1 * scope + 0.1 * assurance
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(mathematical),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
