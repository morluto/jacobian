import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = [
    "EXPONENTIAL_DOMINANCE_USES_ARCHIMEDEAN_ORDER",
    "NO_PROOF_ASSISTANT_REPLAY",
]
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024


def frozen() -> bool:
    visible = W / "input.json"
    frozen_path = T / "input.json"
    try:
        if not (
            is_regular_bounded_file(visible, max_bytes=MAX_INPUT_BYTES)
            and is_regular_bounded_file(frozen_path, max_bytes=MAX_INPUT_BYTES)
        ):
            return False
        return visible.read_bytes() == frozen_path.read_bytes()
    except (OSError, MemoryError):
        return False


def rat(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    n, d = value.get("numerator"), value.get("denominator")
    if type(n) is not int or type(d) is not int or d <= 0:
        return None
    return Fraction(n, d)


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) is int or type(right) is int:
        return type(left) is type(right) and left == right
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def valid(result: object) -> bool:
    keys = {
        "particular_coefficient",
        "homogeneous_base",
        "difference_delta_coefficient",
        "positive_delta_bad_parity",
        "negative_delta_bad_parity",
        "dominance_base",
        "checkpoints",
        "a0",
        "reciprocal",
    }
    if not isinstance(result, dict) or set(result) != keys:
        return False
    if not (
        rat(result["particular_coefficient"]) == Fraction(1, 9)
        and type(result["homogeneous_base"]) is int
        and result["homogeneous_base"] == -7
        and type(result["difference_delta_coefficient"]) is int
        and result["difference_delta_coefficient"] == -8
        and result["positive_delta_bad_parity"] == "EVEN"
        and result["negative_delta_bad_parity"] == "ODD"
        and rat(result["dominance_base"]) == Fraction(7, 2)
        and rat(result["a0"]) == Fraction(1, 9)
        and rat(result["reciprocal"]) == 9
    ):
        return False
    checkpoints = result["checkpoints"]
    if not isinstance(checkpoints, list) or not 4 <= len(checkpoints) <= 10:
        return False
    ns = []
    for item in checkpoints:
        if not isinstance(item, dict) or set(item) != {"n", "a_n", "difference"}:
            return False
        n = item["n"]
        if (
            type(n) is not int
            or not 1 <= n <= 30
            or rat(item["a_n"]) != Fraction(2**n, 9)
            or rat(item["difference"]) != Fraction(2**n, 9)
        ):
            return False
        ns.append(n)
    # Independently replay the closed form and recurrence beyond submitted points.
    for n in range(41):
        if Fraction(2 ** (n + 1), 9) != 2**n - 7 * Fraction(2**n, 9):
            return False
    return len(ns) == len(set(ns))


def main() -> None:
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    evidence = (
        read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/stability-certificate.json",
            max_bytes=MAX_EVIDENCE_BYTES,
        )
        if contract
        else None
    )
    math_ok = bool(
        frozen() and isinstance(submission, dict) and valid(submission.get("result"))
    )
    try:
        result_match = _json_equal(
            evidence.get("result") if evidence else None,
            submission.get("result") if isinstance(submission, dict) else None,
        )
    except RecursionError:
        result_match = False
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence.get("schema_version") == "1"
        and evidence.get("task_id") == expected["task_id"]
        and result_match
        and evidence.get("limitations") == submission.get("limitations")
    )
    scope_ok = bool(
        contract
        and submission.get("scope") == "DECLARED_RECURRENCE_ALL_N_AT_LEAST_ONE"
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
