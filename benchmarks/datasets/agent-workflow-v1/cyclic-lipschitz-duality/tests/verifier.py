import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    strict_submission_contract,
)

MARKS = {11, 23, 35, 47, 59}


def fraction(text):
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return value if str(value) == text else None


def minimum_cost():
    weights = [Fraction(11, 12) if i in MARKS else Fraction(-1, 12) for i in range(60)]
    cumulative = []
    total = Fraction()
    for value in weights:
        total += value
        cumulative.append(total)
    median = sorted(cumulative)[29]
    return sum(abs(value - median) for value in cumulative)


def valid(result):
    try:
        sequence = [fraction(value) for value in result["sequence"]]
        flow = [fraction(value) for value in result["flow"]]
        if any(value is None for value in sequence + flow):
            return False
        weights = [
            Fraction(11, 12) if i in MARKS else Fraction(-1, 12) for i in range(60)
        ]
        return bool(
            len(sequence) == len(flow) == 60
            and sum(sequence) == 0
            and all(abs(sequence[i] - sequence[(i + 1) % 60]) <= 1 for i in range(60))
            and sum(sequence[i] for i in MARKS) == 15
            and all(flow[i] - flow[i - 1] == weights[i] for i in range(60))
            and sum(abs(value) for value in flow) == 15
            and minimum_cost() == 15
            and result["primal_value"] == result["dual_value"] == "15"
        )
    except (KeyError, TypeError):
        return False


def main():
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id="jacobian/cyclic-lipschitz-duality",
        conclusion="OPTIMUM_CERTIFIED",
        verification_record="forbidden",
    )
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(contract and valid(data.get("result", {})))
    evidence_valid = bool(
        contract
        and evidence_list_is_bound(
            data.get("evidence"), expected_path="evidence/answer.txt"
        )
    )
    folded = str(data.get("scope", "")).casefold()
    scope_correct = bool(
        contract
        and "60-entry cyclic sequence" in folded
        and "five marked positions" in folded
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
