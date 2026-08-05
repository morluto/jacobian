import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

E = Path("/tests")
W = Path("/app")
ALLOWED_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED"})


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


def _evidence_matches_result(submission):
    """Bind evidence content to the submitted countermodel.

    The public instruction requires a concise derivation in answer.txt.
    A digest-bound file of unrelated bytes, or the original solution's
    derivation after an alternate countermodel is submitted, must not
    score as valid evidence.  The evidence must carry a RESULT_JSON marker
    whose canonical JSON exactly equals the submitted result.
    """

    if not isinstance(submission, dict):
        return False
    evidence = submission.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not text.strip():
        return False
    markers = [
        line[len("RESULT_JSON:") :].strip()
        for line in text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    if len(markers) != 1:
        return False
    marker = markers[0]
    try:
        parsed = json.loads(marker)
    except (ValueError, RecursionError, MemoryError):
        return False
    if not isinstance(parsed, dict):
        return False
    expected = json.dumps(
        submission.get("result"),
        sort_keys=True,
        separators=(",", ":"),
    )
    actual = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return actual == expected


def main():
    submission = load_submission()
    source = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
        allowed_assurances=ALLOWED_ASSURANCES,
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    # Mathematical correctness is evaluated independently of the envelope and
    # input binding so a protocol, assurance, or input-validity failure is not
    # misreported as wrong mathematics.  Input validity is reported as its own
    # diagnostic and only aggregate reward is gated on it.
    math_correct = _valid_countermodel(result, source)
    input_bound = workspace_input_is_bound()
    evidence_valid = bool(
        isinstance(submission, dict)
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
        and evidence_list_is_bound(
            submission["evidence"], expected_path="evidence/answer.txt"
        )
        and _evidence_matches_result(submission)
    )
    scope_correct = bool(
        isinstance(submission, dict)
        and submission.get("scope") == expected["required_scope"]
        and submission.get("limitations") == expected["limitations"]
    )
    assurance_correct = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    # Aggregate reward is zero for wrong mathematics, false certification,
    # malformed or escaped evidence, or unbound input.  Scope and assurance
    # failures reduce reward but do not zero it, preserving diagnostic
    # independence.
    aggregate_eligible = bool(
        contract
        and math_correct
        and input_bound
        and evidence_valid
        and not false_certification
    )
    reward = (
        0.0
        if not aggregate_eligible
        else 0.7 + 0.1 * evidence_valid + 0.1 * scope_correct + 0.1 * assurance_correct
    )
    output = Path("/logs/verifier/reward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_bound),
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
