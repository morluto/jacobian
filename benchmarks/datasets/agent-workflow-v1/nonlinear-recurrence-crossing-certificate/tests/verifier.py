import json
import math
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")
TERMINAL = {
    ("0<a<2", "a_next<3/2"),
    ("0<a<1", "a_next<0"),
    ("1<a<3/2", "0<a_next<5/6<1"),
}


def _frozen():
    try:
        raw = (E / "input.json").read_bytes()
        if (W / "input.json").is_symlink() or (W / "input.json").read_bytes() != raw:
            return {}
        return json.loads(raw)
    except (OSError, ValueError):
        return {}


def _result_valid(result, frozen):
    required = {
        "potential_recurrence",
        "initial_potential",
        "threshold",
        "decrement_lower_bound",
        "phase_transitions",
        "threshold_index_upper",
        "terminal_bounds",
        "negative_index_upper",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    rat = result["decrement_lower_bound"]
    if (
        not isinstance(rat, dict)
        or set(rat) != {"numerator", "denominator"}
        or any(type(rat[k]) is not int for k in rat)
    ):
        return False
    bounds = result["terminal_bounds"]
    parsed = (
        {(x.get("premise"), x.get("conclusion")) for x in bounds}
        if isinstance(bounds, list)
        and all(
            isinstance(x, dict) and set(x) == {"premise", "conclusion"} for x in bounds
        )
        else set()
    )
    transitions = result["phase_transitions"]
    exact_phase = (
        3136 * 4 - transitions * 7 < 4 * 4 and 3136 * 4 - (transitions - 1) * 7 >= 4 * 4
    )
    return bool(
        frozen.get("initial_value") == 56
        and result["potential_recurrence"] == "d[n+1]=d[n]-2+1/d[n]"
        and result["initial_potential"] == 56**2
        and result["threshold"] == 4
        and rat == {"numerator": 7, "denominator": 4}
        and math.gcd(rat["numerator"], rat["denominator"]) == 1
        and type(transitions) is int
        and exact_phase
        and result["threshold_index_upper"] == 1 + transitions
        and parsed == TERMINAL
        and len(bounds) == 3
        and result["negative_index_upper"] == result["threshold_index_upper"] + 3
        and result["negative_index_upper"] < frozen["target"]["index_upper_exclusive"]
    )


def main():
    submission, frozen = load_submission(), _frozen()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _result_valid(submission.get("result"), frozen))
    evidence = None
    if (
        contract
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
    ):
        evidence = read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/nonlinear-recurrence-certificate.json",
        )
    evidence_valid = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == expected["task_id"]
        and evidence["result"] == submission.get("result")
        and evidence["limitations"] == submission.get("limitations")
    )
    envelope = bool(
        contract
        and submission.get("scope") == "TOTAL_REAL_RECURRENCE_FROM_A1_EQUALS_56"
        and submission.get("limitations") == ["ELEMENTARY_REAL_ORDER_LEMMAS_TRUSTED"]
    )
    assurance = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        math_correct and evidence_valid and envelope and not false_certification
    )
    reward = 0 if not correct else 0.9 + 0.1 * assurance
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(envelope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
