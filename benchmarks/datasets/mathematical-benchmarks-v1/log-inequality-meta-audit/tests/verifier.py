import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")


def _valid_layered_audit(result, source):
    if not isinstance(result, dict) or set(result) != {
        "claim_status",
        "counterexample_status",
        "instruction_compliance",
        "evaluator_score",
        "evaluator_rubric_status",
        "meta_evaluation_status",
        "certificate",
    }:
        return False
    certificate = result["certificate"]
    if not isinstance(certificate, dict) or set(certificate) != {
        "x_squared",
        "y_squared",
        "z_squared",
        "first_log_term",
        "lhs_reduction",
        "comparison_left",
        "comparison_right",
        "comparison",
    }:
        return False
    if not all(
        type(certificate[key]) is int
        for key in (
            "x_squared",
            "y_squared",
            "z_squared",
            "first_log_term",
            "comparison_left",
            "comparison_right",
        )
    ):
        return False

    witness = source["conversation_trace"]["proposed_solution"]["witness"]
    domain_valid = bool(
        certificate["x_squared"] == witness["x_squared"] == 6
        and certificate["y_squared"] == witness["y_squared"] == 6
        and certificate["z_squared"] == witness["z_squared"] == 12
        and certificate["x_squared"] + certificate["y_squared"]
        == certificate["z_squared"]
        and min(
            certificate["x_squared"],
            certificate["y_squared"],
            certificate["z_squared"],
        )
        > 1
    )
    left = 18**4
    right = 6**7
    exact_disproof = bool(
        certificate["first_log_term"] == 2
        and certificate["lhs_reduction"] == "2+4*log_6(18)"
        and certificate["comparison_left"] == left
        and certificate["comparison_right"] == right
        and certificate["comparison"] == "LESS_THAN"
        and left < right
    )
    rubric = source["evaluation_rubric"]
    observed_evaluation = source["conversation_trace"]["solution_evaluation"]
    meta = source["conversation_trace"]["meta_evaluation"]
    expected_score = (
        rubric["score_if_noncompliant"]
        if result["instruction_compliance"] == rubric["noncompliant_instruction_label"]
        else rubric["score_if_compliant"]
    )
    evaluator_consistent = (
        type(result["evaluator_score"]) is int
        and result["evaluator_score"] == expected_score
        and observed_evaluation["score"] == expected_score
        and result["evaluator_rubric_status"] == rubric["consistent_status"]
        and observed_evaluation["mathematical_reasoning"]
        == rubric["mathematical_reasoning_label"]
    )
    return bool(
        domain_valid
        and exact_disproof
        and result["claim_status"] == "FALSE"
        and result["counterexample_status"] == "VALID"
        and result["instruction_compliance"] == "DOES_NOT_PROVE_REQUESTED_CLAIM"
        and result["instruction_compliance"]
        in {
            rubric["compliant_instruction_label"],
            rubric["noncompliant_instruction_label"],
        }
        and evaluator_consistent
        and result["meta_evaluation_status"] == rubric["reasonable_meta_status"]
        and meta["rating"] == rubric["reasonable_meta_rating"]
    )


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    source = json.loads(next(E.glob("*input*.json")).read_text())
    math_correct = bool(
        protocol_ok and _valid_layered_audit(submission.get("result"), source)
    )
    reward = aggregate_reward(
        correctness=math_correct,
        witness_validity=True,
        protocol_ok=protocol_ok,
    )

    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
