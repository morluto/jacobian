import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

W = Path("/app")
T = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
MAX_INPUT_BYTES = 1_048_576
LIMITATIONS = [
    "MATRIX_DETERMINANT_LEMMA_TRUSTED",
    "RATIONAL_LIMIT_INFERENCE_NOT_PROOF_ASSISTANT_VERIFIED",
]
ROOT_FORMULA = "4*n*(n+1)/((n-1)*(n+2))"


def frozen_contract() -> dict:
    """Bind the agent-visible input to the frozen verifier copy before parsing.

    Reject oversized or non-regular workspace input so a hostile replacement
    cannot OOM or block the verifier before a deterministic reward is written.
    """
    try:
        a, t = W / "input.json", T / "input.json"
        if not workspace_input_is_bound(
            a, tests=T
        ) or not is_regular_bounded_file(a, max_bytes=MAX_INPUT_BYTES):
            return {}
        raw = t.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError, RecursionError, MemoryError):
        return {}
    return (
        value
        if isinstance(value, dict)
        and value.get("task_id") == "jacobian/rank-one-spectral-limit-certificate"
        else {}
    )


def rat(value: object) -> Fraction | None:
    """Parse a rational as ``{"numerator": int, "denominator": positive int}``.

    Accept any mathematically equivalent encoding (e.g. ``2/4``); ``Fraction``
    normalizes the value so reduced form is not required for rational fields.
    """
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    n, d = value.get("numerator"), value.get("denominator")
    if type(n) is not int or type(d) is not int or d <= 0:
        return None
    return Fraction(n, d)


def _json_equal(submitted: object, evidence: object) -> bool:
    """Compare two JSON values preserving exact integer/float distinction.

    Python treats ``True == 1`` and ``4.0 == 4`` as equal, but the public
    schema requires integers; reject numerically equal but type-mismatched
    values so the evidence artifact contains the identical schema-valid result.
    """
    if isinstance(submitted, bool) or isinstance(evidence, bool):
        return submitted is evidence
    if isinstance(submitted, int) and isinstance(evidence, int):
        return submitted == evidence
    if isinstance(submitted, float) or isinstance(evidence, float):
        return False
    if isinstance(submitted, dict) and isinstance(evidence, dict):
        if set(submitted) != set(evidence):
            return False
        return all(_json_equal(submitted[k], evidence[k]) for k in submitted)
    if isinstance(submitted, list) and isinstance(evidence, list):
        return len(submitted) == len(evidence) and all(
            _json_equal(a, b) for a, b in zip(submitted, evidence)
        )
    return submitted == evidence


def certificate_valid(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "rank_one_sign",
        "partial_fraction",
        "checkpoints",
        "root_formula",
        "limit",
    }:
        return False
    pf = result["partial_fraction"]
    if not isinstance(pf, dict) or set(pf) != {
        "scale",
        "left_coefficient",
        "right_coefficient",
    }:
        return False
    scale = rat(pf["scale"])
    left = pf["left_coefficient"]
    right = pf["right_coefficient"]
    if scale is None or type(left) is not int or type(right) is not int:
        return False
    # Validate the partial-fraction identity using the submitted scale and
    # coefficients, accepting any equivalent factorization (e.g. scale=1/6
    # with coefficients 3 and -3) rather than a single normalization.
    for k in range(2, 51):
        proposed = scale * (
            Fraction(left, 1) * Fraction(1, k * (k - 1))
            + Fraction(right, 1) * Fraction(1, k * (k + 1))
        )
        if proposed != Fraction(1, k**3 - k):
            return False
    checkpoints = result["checkpoints"]
    if not isinstance(checkpoints, list) or not 3 <= len(checkpoints) <= 8:
        return False
    ns = []
    for item in checkpoints:
        if not isinstance(item, dict) or set(item) != {"n", "reciprocal_sum", "root"}:
            return False
        n = item["n"]
        if type(n) is not int or not 4 <= n <= 30:
            return False
        expected_sum = sum(
            (Fraction(1, k**3 - k) for k in range(2, n + 1)), Fraction()
        )
        if (
            rat(item["reciprocal_sum"]) != expected_sum
            or rat(item["root"]) != 1 / expected_sum
        ):
            return False
        ns.append(n)
    # Accept distinct checkpoints in any order; the instruction requires only
    # distinct values and the schema declares no ordering constraint.
    return bool(
        len(set(ns)) == len(ns)
        and result["rank_one_sign"] == "DIAGONAL_MINUS_LAMBDA_ONES"
        and result["root_formula"] == ROOT_FORMULA
        and rat(result["limit"]) == 4
    )


def main() -> None:
    frozen = frozen_contract()
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    evidence = None
    if contract:
        evidence_path = W / "evidence" / "spectral-certificate.json"
        if is_regular_bounded_file(evidence_path, max_bytes=MAX_EVIDENCE_BYTES):
            evidence = read_evidence_json(
                submission["evidence"][0],
                expected_path="evidence/spectral-certificate.json",
                max_bytes=MAX_EVIDENCE_BYTES,
            )
    math_ok = bool(contract and frozen and certificate_valid(submission.get("result")))
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence.get("schema_version") == "1"
        and evidence.get("task_id") == expected["task_id"]
        and _json_equal(evidence.get("result"), submission.get("result"))
        and evidence.get("limitations") == LIMITATIONS
    )
    scope_ok = bool(
        contract
        and submission.get("scope") == "DECLARED_MATRIX_FAMILY"
        and submission.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(contract and submission.get("claimed_assurance") == "COMPUTED")
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    correct = math_ok and evidence_ok and scope_ok and not false_cert
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(contract),
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
