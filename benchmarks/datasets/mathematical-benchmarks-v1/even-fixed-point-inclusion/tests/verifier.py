import itertools
import json
import math
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = ["FINITE_PERMUTATIONS_OF_SIZE_8", "NO_GENERAL_ROOK_POLYNOMIAL_PROOF"]


def derive():
    histogram = [0] * 5
    for permutation in itertools.permutations(range(1, 9)):
        fixed = sum(permutation[value - 1] == value for value in (2, 4, 6, 8))
        histogram[fixed] += 1
    terms = [(-1) ** j * math.comb(4, j) * math.factorial(8 - j) for j in range(5)]
    return {
        "signed_inclusion_terms": terms,
        "inclusion_sum": sum(terms),
        "exact_even_fixed_histogram": histogram,
    }


def matches(result):
    return result == derive()


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def main():
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    evidence = (
        read_evidence_json(
            submission["evidence"][0], expected_path="evidence/answer.txt"
        )
        if contract
        else None
    )
    derived = derive()
    math_ok = bool(contract and frozen() and matches(submission.get("result")))
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence.get("schema_version") == "1"
        and evidence.get("task_id") == expected["task_id"]
        and evidence.get("result") == derived
        and evidence.get("limitations") == LIMITATIONS
    )
    scope_ok = bool(
        contract
        and submission.get("scope")
        == "ALL_8_FACTORIAL_PERMUTATIONS_AND_FIVE_INCLUSION_TERMS"
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == LIMITATIONS
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
