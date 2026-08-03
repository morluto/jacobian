import itertools
import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    strict_submission_contract,
)


def valid_family(n, raw_family):
    try:
        family = [tuple(item) for item in raw_family]
        if len(family) != len(set(family)):
            return False
        if any(
            tuple(sorted(item)) != item or len(item) != len(set(item))
            for item in family
        ):
            return False
        if any(value < 0 or value >= n for item in family for value in item):
            return False
        if len(family) != 1 + n + n // 2:
            return False
        sets = [set(item) for item in family]
        if any(
            first & second & third
            for first, second, third in itertools.combinations(sets, 3)
        ):
            return False
        return all(sum(value in item for item in sets) <= 2 for value in range(n))
    except (TypeError, ValueError):
        return False


def main():
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id="jacobian/triplewise-empty-extremal-audit",
        conclusion="SOURCE_BOUND_REPAIRED",
        verification_record="forbidden",
    )
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result", {})
    math_correct = False
    try:
        certificate = result["upper_bound_certificate"]
        constructions = {item["n"]: item["family"] for item in result["constructions"]}
        math_correct = bool(
            contract
            and result["maximum_formula"] == "1+n+floor(n/2)"
            and certificate
            == {
                "element_frequency_cap": 2,
                "singleton_cap": "n",
                "nonsingleton_incidence_floor": 2,
            }
            and set(constructions) == {7, 8, 11}
            and all(valid_family(n, constructions[n]) for n in constructions)
        )
    except (KeyError, TypeError):
        pass
    evidence_valid = bool(
        contract
        and evidence_list_is_bound(
            data.get("evidence"), expected_path="evidence/answer.txt"
        )
    )
    folded = str(data.get("scope", "")).casefold()
    scope_correct = bool(
        contract
        and "distinct subsets" in folded
        and "triplewise empty intersection" in folded
    )
    assurance_correct = bool(contract and data.get("claimed_assurance") == "COMPUTED")
    false_certification = bool(
        false_verified_claim(submission, verification_record_bound=False)
    )
    reward = (
        1.0
        if math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and not false_certification
        else 0.0
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
