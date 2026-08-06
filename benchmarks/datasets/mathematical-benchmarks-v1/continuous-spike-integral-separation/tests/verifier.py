import json
import re
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from verifier_support import (
    ASSURANCE_LEVELS,
    SUBMISSION_FIELDS,
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = [
    "STANDARD_HARMONIC_AND_P_SERIES_CRITERIA_TRUSTED",
    "TWELVE_EXACT_SPIKES_PLUS_GENERAL_FORMULA",
    "NO_PROOF_ASSISTANT_VERIFICATION",
]
ALLOWED_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED"})
MAX_RATIONAL_LEN = 100
EVIDENCE_KEYWORDS = (
    "disjoint",
    "integer",
    "diverg",
    "converg",
)


def fraction(value):
    if not isinstance(value, str) or len(value) > MAX_RATIONAL_LEN:
        raise ValueError
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value) is None:
        raise ValueError
    parsed = Fraction(value)
    if str(parsed) != value:
        raise ValueError
    return parsed


def expected_spike(n, alpha):
    center = Fraction(2 * n + 1, 2)
    width = alpha / n
    return {
        "n": n,
        "center": str(center),
        "half_width": str(width),
        "left": str(center - width),
        "right": str(center + width),
        "area": str(width),
        "integer_sample": str(Fraction(1, n * n)),
    }


def _valid_spikes(spikes, alpha):
    if not isinstance(spikes, list) or len(spikes) != 12:
        return False
    if any(
        not isinstance(spike, dict) or type(spike.get("n")) is not int
        for spike in spikes
    ):
        return False
    expected = {n: expected_spike(n, alpha) for n in range(1, 13)}
    by_n = {}
    for spike in spikes:
        n = spike["n"]
        if n in by_n or spike != expected.get(n):
            return False
        by_n[n] = spike
    if set(by_n) != set(expected):
        return False
    ordered = [by_n[n] for n in sorted(by_n)]
    for left, right in pairwise(ordered):
        if fraction(left["right"]) >= fraction(right["left"]):
            return False
    for spike in spikes:
        left, right = fraction(spike["left"]), fraction(spike["right"])
        n = spike["n"]
        if left <= n or right >= n + 1:
            return False
    return True


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "alpha",
        "baseline_power",
        "spike_height",
        "spikes",
        "integral_classification",
        "sample_series_classification",
    }:
        return False
    try:
        alpha = fraction(result["alpha"])
    except (ValueError, ZeroDivisionError):
        return False
    if not (0 < alpha <= Fraction(1, 4)):
        return False
    if result["baseline_power"] != 2 or result["spike_height"] != "1":
        return False
    if not _valid_spikes(result.get("spikes"), alpha):
        return False
    return bool(
        result["integral_classification"]
        == {"spike_area_series": "alpha*sum(1/n)", "status": "DIVERGENT"}
        and result["sample_series_classification"]
        == {"sample_series": "sum(1/n^2)", "status": "CONVERGENT"}
    )


def structural_contract(submission, *, task_id, conclusion):
    """Assurance-neutral envelope check for math/evidence/scope dimensions.

    Accepts any well-formed assurance level (including an overclaimed
    ``VERIFIED``) so that a false certification zeros only the assurance
    calibration dimension and the aggregate reward, not the independently
    computed mathematical correctness, evidence validity, or scope accuracy.
    """

    if not isinstance(submission, dict):
        return False
    verified = submission.get("claimed_assurance") == "VERIFIED"
    fields = frozenset(SUBMISSION_FIELDS)
    if verified:
        fields = frozenset(SUBMISSION_FIELDS | {"verification_record_uri"})
    return bool(
        frozenset(submission) == fields
        and submission.get("task_id") == task_id
        and submission.get("conclusion") == conclusion
        and submission.get("completeness") == "COMPLETE_FOR_DECLARED_FAMILY"
        and isinstance(submission.get("result"), dict)
        and isinstance(submission.get("scope"), str)
        and isinstance(submission.get("limitations"), list)
        and all(type(item) is str for item in submission.get("limitations", []))
        and isinstance(submission.get("evidence"), list)
        and len(submission.get("evidence", [])) == 1
        and isinstance(submission.get("claimed_assurance"), str)
        and submission.get("claimed_assurance") in ASSURANCE_LEVELS
    )


def evidence_content_valid(submission):
    """Reject empty or unrelated certificate evidence.

    The public instruction requires the certificate to show support
    disjointness, integer avoidance, and the two series arguments. A
    digest-bound file of unrelated or empty bytes must not score as valid
    evidence even though its path and hash match.
    """

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
    lowered = text.lower()
    return all(keyword in lowered for keyword in EVIDENCE_KEYWORDS)


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
    structural = structural_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
    )
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        completeness="COMPLETE_FOR_DECLARED_FAMILY",
        allowed_assurances=ALLOWED_ASSURANCES,
        verification_record="forbidden",
    )
    math_ok = bool(structural and frozen() and valid_result(submission.get("result")))
    evidence_bound = bool(
        structural
        and evidence_list_is_bound(
            submission.get("evidence"), expected_path="evidence/answer.txt"
        )
    )
    evidence_ok = bool(evidence_bound and evidence_content_valid(submission))
    scope_ok = bool(
        structural
        and submission.get("scope") == expected["required_scope"]
        and submission.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        contract and submission.get("claimed_assurance") in ALLOWED_ASSURANCES
    )
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
