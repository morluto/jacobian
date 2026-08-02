import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")


def _fraction(text):
    if not isinstance(text, str):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if str(value) == text else None


def _model(value, *, intended, proposed, bound, lower, upper):
    if not isinstance(value, dict) or set(value) != {
        "limsup_values",
        "intended_truth",
        "proposed_truth",
        "distinguishing_index",
    }:
        return False
    raw = value["limsup_values"]
    if not isinstance(raw, list) or not 2 <= len(raw) <= 8:
        return False
    values = [_fraction(item) for item in raw]
    if any(item is None or item < lower or item > upper for item in values):
        return False
    intended_truth = any(item <= bound for item in values)
    proposed_truth = all(item >= bound for item in values)
    index = value["distinguishing_index"]
    if type(index) is not int or not 0 <= index < len(values):
        return False
    distinguishing = values[index]
    index_valid = distinguishing <= bound if intended else distinguishing < bound
    if proposed and not intended:
        index_valid = distinguishing > bound
    return bool(
        value["intended_truth"] is intended_truth is intended
        and value["proposed_truth"] is proposed_truth is proposed
        and index_valid
    )


def _valid_result(result, source):
    if not isinstance(result, dict) or set(result) != {
        "relationship",
        "intended_only_model",
        "proposed_only_model",
    }:
        return False
    bound = Fraction(source["bound"])
    limits = source["model_constraints"]
    lower, upper = Fraction(limits["minimum_value"]), Fraction(limits["maximum_value"])
    return bool(
        result["relationship"] == "INCOMPARABLE"
        and _model(
            result["intended_only_model"],
            intended=True,
            proposed=False,
            bound=bound,
            lower=lower,
            upper=upper,
        )
        and _model(
            result["proposed_only_model"],
            intended=False,
            proposed=True,
            bound=bound,
            lower=lower,
            upper=upper,
        )
    )


def _evidence_valid(evidence):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text().casefold()
    except (OSError, UnicodeError):
        return False
    return all(word in text for word in ("existential", "universal", "incomparable"))


def main():
    submission = load_submission()
    source = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _valid_result(submission.get("result"), source))
    evidence_valid = bool(contract and _evidence_valid(submission.get("evidence")))
    scope_correct = bool(
        contract and submission.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    limitations_correct = bool(
        contract
        and any(
            "open problem" in item.casefold()
            for item in submission.get("limitations", [])
            if isinstance(item, str)
        )
    )
    correct = bool(
        contract and math_correct and limitations_correct and not false_certification
    )
    reward = (
        0
        if not correct
        else 0.7 + 0.1 * evidence_valid + 0.1 * scope_correct + 0.1 * assurance_correct
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
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
