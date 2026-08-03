import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
LIMITATION = (
    "The checker validates exact directional series coefficients and the "
    "declared audit; it is not a general asymptotic-analysis prover or "
    "external proof assistant."
)


def _load_frozen_input() -> dict[str, Any]:
    try:
        workspace = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _rational(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or not -1_000_000 <= numerator <= 1_000_000
        or not 1 <= denominator <= 1_000_000
    ):
        return None
    if math.gcd(abs(numerator), denominator) != 1:
        return None
    return Fraction(numerator, denominator)


def _directional_witness_is_valid(
    witness: object,
) -> tuple[str, tuple[Fraction, ...]] | None:
    if not isinstance(witness, dict) or set(witness) != {
        "direction",
        "quadratic_coefficient",
        "sign",
        "normalized_residual_limit",
    }:
        return None
    direction_raw = witness["direction"]
    if not isinstance(direction_raw, list) or not 2 <= len(direction_raw) <= 4:
        return None
    direction = tuple(_rational(item) for item in direction_raw)
    if any(value is None for value in direction):
        return None
    exact = tuple(value for value in direction if value is not None)
    if sum(value * value for value in exact) != 1:
        return None

    # Normalize x to e_1 and y=t*u.  Expanding
    # (1 - 2*u_1*t + t^2)^(-1/2) through t^2 gives
    # 1 + u_1*t + (3*u_1^2 - 1)t^2/2 + O(t^3).
    coefficient = (3 * exact[0] * exact[0] - 1) / 2
    submitted = _rational(witness["quadratic_coefficient"])
    sign = witness["sign"]
    expected_sign = "POSITIVE" if coefficient > 0 else "NEGATIVE"
    if (
        coefficient == 0
        or submitted != coefficient
        or sign != expected_sign
        or witness["normalized_residual_limit"] != "quadratic_coefficient"
    ):
        return None
    return expected_sign, exact


def _result_is_valid(result: object, source: dict[str, Any]) -> bool:
    if (
        source.get("source", {}).get("revision")
        != "f5935720f176cedff4ecd8ebf83d1696e31cfac8"
        or source.get("source", {}).get("row") != 5001
        or source.get("source", {}).get("source_id") != 43363
        or source.get("source", {}).get("row_sha256")
        != "sha256:c5bfe234c517c99357fbabc3325bb1289829822aa3db7908ff40a9e191e76497"
        or source.get("claim", {}).get("dataset_label") is not False
        or source.get("audit_contract", {}).get("normalization")
        != "by rotational invariance set x=e_1 and y=t*u with |u|=1"
    ):
        return False
    if not isinstance(result, dict) or set(result) != {
        "claim_status",
        "second_order_term",
        "directional_witnesses",
        "response_audit",
    }:
        return False
    if result["claim_status"] != "FALSE":
        return False

    second = result["second_order_term"]
    if not isinstance(second, dict) or set(second) != {
        "dot_square_coefficient",
        "norm_square_coefficient",
        "denominator_power",
        "remainder",
    }:
        return False
    if (
        _rational(second["dot_square_coefficient"]) != Fraction(3, 2)
        or _rational(second["norm_square_coefficient"]) != Fraction(-1, 2)
        or second["denominator_power"] != 5
        or second["remainder"] != "O(|y|^3/|x|^4)"
    ):
        return False

    witnesses = result["directional_witnesses"]
    if not isinstance(witnesses, list) or len(witnesses) != 2:
        return False
    checked = [_directional_witness_is_valid(item) for item in witnesses]
    if any(item is None for item in checked):
        return False
    valid = [item for item in checked if item is not None]
    if {item[0] for item in valid} != {"POSITIVE", "NEGATIVE"}:
        return False
    if valid[0][1] == valid[1][1]:
        return False

    audit = result["response_audit"]
    defects = {
        "EPSILON_SQUARED_ORDER_MISCLASSIFIED",
        "EXPLICIT_QUADRATIC_TERM_DROPPED",
        "CUBIC_REMAINDER_FALSE",
    }
    return bool(
        isinstance(audit, dict)
        and set(audit) == {"conclusion", "defects"}
        and audit["conclusion"] == "RIGHT_CONCLUSION_WITH_INVALID_ORDER_STEP"
        and isinstance(audit["defects"], list)
        and len(audit["defects"]) == 3
        and set(audit["defects"]) == defects
    )


def _evidence_matches_result(evidence: object, result: dict[str, Any]) -> bool:
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(evidence, list) or not evidence:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        if target.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        text = target.read_text().casefold()
        return all(
            fragment in text
            for fragment in (
                "3*u_1^2-1",
                "positive quadratic coefficient 1",
                "negative coefficient -1/2",
                "epsilon^2",
                "quadratic contribution",
            )
        )
    except (OSError, UnicodeError):
        return False


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    source = _load_frozen_input()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = data.get("result")
    math_correct = bool(contract and _result_is_valid(result, source))
    evidence_valid = bool(
        math_correct
        and isinstance(result, dict)
        and _evidence_matches_result(data.get("evidence"), result)
    )
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(contract and LIMITATION in data.get("limitations", []))
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
