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
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = [
    "STANDARD_POWER_LOG_INTEGRAL_CRITERION_TRUSTED",
    "DECLARED_FUNCTION_FAMILY_ONLY",
    "NO_PROOF_ASSISTANT_VERIFICATION",
]
MAX_EVIDENCE_BYTES = 64 * 1024


def fraction(value):
    """Parse a rational from a string, accepting mathematically equivalent forms."""
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value) is None:
        raise ValueError
    return Fraction(value)


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "beta",
        "origin_power_coefficient",
        "infinity_power_coefficient",
        "p2_log_exponent",
        "p2_integral_each",
        "critical_p",
        "lower_regime",
        "upper_regime",
    }:
        return False
    try:
        beta = fraction(result["beta"])
        log_exponent = fraction(result["p2_log_exponent"])
        integral = fraction(result["p2_integral_each"])
    except (ValueError, ZeroDivisionError):
        return False
    return bool(
        beta > Fraction(1, 2)
        and result["origin_power_coefficient"] == "-1/2"
        and result["infinity_power_coefficient"] == "-1/2"
        and log_exponent == -2 * beta
        and integral == 1 / (2 * beta - 1)
        and result["critical_p"] == "2"
        and result["lower_regime"]
        == {"p_interval": "0<p<2", "obstruction": "INFINITY_POWER_TAIL"}
        and result["upper_regime"]
        == {"p_interval": "p>2", "obstruction": "ORIGIN_POWER_SINGULARITY"}
    )


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def _json_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def valid_evidence(evidence, result):
    """Bind evidence content to the submitted result via a RESULT_JSON line."""
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    if not evidence_list_is_bound(
        evidence, expected_path="evidence/answer.txt", max_bytes=MAX_EVIDENCE_BYTES
    ):
        return False
    target = resolve_evidence(
        evidence[0], expected_path="evidence/answer.txt", max_bytes=MAX_EVIDENCE_BYTES
    )
    if target is None:
        return False
    try:
        text = target.read_text()
    except (OSError, UnicodeError):
        return False
    markers = [
        line.removeprefix("RESULT_JSON:").strip()
        for line in text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    if len(markers) != 1:
        return False
    try:
        bound_result = json.loads(markers[0])
    except (ValueError, RecursionError):
        return False
    if not isinstance(result, dict):
        return False
    return _json_equal(bound_result, result)


def main():
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        completeness="COMPLETE_FOR_DECLARED_FAMILY",
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    envelope = isinstance(submission, dict)
    result = submission.get("result") if envelope else None
    # Evaluate mathematical correctness independently of the assurance/contract
    # envelope so an unsupported assurance claim does not collapse the
    # correctness diagnostic to zero for an otherwise valid result.
    math_ok = bool(envelope and frozen() and valid_result(result))
    evidence_ok = bool(
        envelope and frozen() and valid_evidence(submission.get("evidence"), result)
    )
    scope_ok = bool(
        envelope
        and submission.get("scope") == expected["required_scope"]
        and submission.get("completeness") == "COMPLETE_FOR_DECLARED_FAMILY"
        and submission.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        envelope
        and isinstance(submission.get("claimed_assurance"), str)
        and submission.get("claimed_assurance") == "COMPUTED"
    )
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    correct = bool(contract and math_ok and scope_ok and evidence_ok and not false_cert)
    reward = 0 if not correct else 0.9 + 0.1 * assurance_ok
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": reward,
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
