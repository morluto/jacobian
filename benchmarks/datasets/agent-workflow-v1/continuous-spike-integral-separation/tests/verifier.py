import json
import re
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    strict_submission_contract,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = [
    "STANDARD_HARMONIC_AND_P_SERIES_CRITERIA_TRUSTED",
    "TWELVE_EXACT_SPIKES_PLUS_GENERAL_FORMULA",
    "NO_PROOF_ASSISTANT_VERIFICATION",
]


def fraction(value):
    if not isinstance(value, str) or len(value) > 100:
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
    spikes = result.get("spikes")
    if not isinstance(spikes, list) or len(spikes) != 12:
        return False
    if any(
        not isinstance(spike, dict) or type(spike.get("n")) is not int
        for spike in spikes
    ):
        return False
    if spikes != [expected_spike(n, alpha) for n in range(1, 13)]:
        return False
    for left, right in pairwise(spikes):
        if fraction(left["right"]) >= fraction(right["left"]):
            return False
    for spike in spikes:
        left, right = fraction(spike["left"]), fraction(spike["right"])
        n = spike["n"]
        if left <= n or right >= n + 1:
            return False
    return bool(
        result["integral_classification"]
        == {"spike_area_series": "alpha*sum(1/n)", "status": "DIVERGENT"}
        and result["sample_series_classification"]
        == {"sample_series": "sum(1/n^2)", "status": "CONVERGENT"}
    )


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
        completeness="COMPLETE_FOR_DECLARED_FAMILY",
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    math_ok = bool(contract and frozen() and valid_result(submission.get("result")))
    evidence_ok = bool(
        contract
        and evidence_list_is_bound(
            submission.get("evidence"), expected_path="evidence/answer.txt"
        )
    )
    scope_ok = bool(
        contract
        and submission.get("scope") == expected["required_scope"]
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
