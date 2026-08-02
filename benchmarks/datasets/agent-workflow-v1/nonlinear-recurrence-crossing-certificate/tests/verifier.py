import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")
NEG_INF, POS_INF = "NEGATIVE_INFINITY", "POSITIVE_INFINITY"


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
        "potential_identity_coefficients",
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

    def rational(value):
        if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
            return None
        numerator, denominator = value["numerator"], value["denominator"]
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator < 1
        ):
            return None
        if math.gcd(numerator, denominator) != 1:
            return None
        return Fraction(numerator, denominator)

    bounds = result["terminal_bounds"]
    parsed = set()
    if isinstance(bounds, list) and len(bounds) == 3:
        for item in bounds:
            if not isinstance(item, dict) or set(item) != {
                "input_lower",
                "input_upper",
                "output_lower",
                "output_upper",
            }:
                return False
            values = []
            for key in ("input_lower", "input_upper", "output_lower", "output_upper"):
                raw = item[key]
                values.append(raw if raw in (NEG_INF, POS_INF) else rational(raw))
            if any(value is None for value in values):
                return False
            parsed.add(tuple(values))
    expected_bounds = {
        (Fraction(0), Fraction(2), NEG_INF, Fraction(3, 2)),
        (Fraction(0), Fraction(1), NEG_INF, Fraction(0)),
        (Fraction(1), Fraction(3, 2), Fraction(0), Fraction(5, 6)),
    }
    transitions = result["phase_transitions"]
    exact_phase = (
        3136 * 4 - transitions * 7 < 4 * 4 and 3136 * 4 - (transitions - 1) * 7 >= 4 * 4
    )
    return bool(
        frozen.get("initial_value") == 56
        and result["potential_identity_coefficients"] == [1, -2, 1]
        and result["initial_potential"] == 56**2
        and rational(result["threshold"]) == 4
        and rat == {"numerator": 7, "denominator": 4}
        and math.gcd(rat["numerator"], rat["denominator"]) == 1
        and type(transitions) is int
        and exact_phase
        and result["threshold_index_upper"] == 1 + transitions
        and parsed == expected_bounds
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
