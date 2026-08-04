import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    strict_submission_contract,
)

E = Path("/tests")


def _fraction(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value) is None:
        raise ValueError
    parsed = Fraction(value)
    if str(parsed) != value:
        raise ValueError
    return parsed


def _valid_countermodel(result, source):
    keys = {
        "left_slope",
        "right_slope",
        "offset",
        "jump",
        "left_endpoint_value",
        "left_limit",
        "right_breakpoint_value",
        "right_endpoint_value",
        "gap_witness",
    }
    if not isinstance(result, dict) or set(result) != keys:
        return False
    try:
        value = {key: _fraction(item) for key, item in result.items()}
        bounds = source["parameter_bounds"]
        left = _fraction(source["interval"]["left"])
        right = _fraction(source["interval"]["right"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    for key in ("left_slope", "right_slope", "jump", "offset"):
        try:
            if (
                not _fraction(bounds[key]["minimum"])
                <= value[key]
                <= _fraction(bounds[key]["maximum"])
            ):
                return False
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False
    m_left = value["left_slope"]
    m_right = value["right_slope"]
    offset = value["offset"]
    jump = value["jump"]
    left_limit = offset
    right_zero = offset + jump
    witness = value["gap_witness"]
    return bool(
        m_left > 0
        and m_right > 0
        and jump > 0
        and value["left_endpoint_value"] == m_left * left + offset
        and value["left_limit"] == left_limit
        and value["right_breakpoint_value"] == right_zero
        and value["right_endpoint_value"] == m_right * right + right_zero
        and left_limit < witness < right_zero
        and value["left_endpoint_value"] < witness < value["right_endpoint_value"]
    )


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
    math_correct = bool(
        contract and _valid_countermodel(submission.get("result"), source)
    )
    evidence_valid = bool(
        contract
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
        and evidence_list_is_bound(
            submission["evidence"], expected_path="evidence/answer.txt"
        )
    )
    scope_correct = bool(
        contract
        and submission.get("scope") == expected["required_scope"]
        and submission.get("limitations") == expected["limitations"]
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(contract and math_correct and not false_certification)
    reward = (
        0
        if not correct
        else 0.7 + 0.1 * evidence_valid + 0.1 * scope_correct + 0.1 * assurance_correct
    )
    output = Path("/logs/verifier/reward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
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
