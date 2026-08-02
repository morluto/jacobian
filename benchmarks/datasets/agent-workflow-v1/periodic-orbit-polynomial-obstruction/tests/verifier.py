import json
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")


def _load_frozen_input():
    try:
        raw = (E / "input.json").read_bytes()
        if (
            (W / "input.json").is_symlink()
            or (E / "input.json").is_symlink()
            or (W / "input.json").read_bytes() != raw
        ):
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _reduction(item):
    if not isinstance(item, dict) or set(item) != {
        "modulus",
        "residue_basis",
        "residue_coefficients",
        "conclusion",
    }:
        return None
    modulus = item["modulus"]
    expected = {
        "p": (["P(q)", "P(1)"], [-1, 1], "p_DIVIDES_P(q)-P(1)"),
        "q": (["P(p)", "P(1)"], [-1, 1], "q_DIVIDES_P(p)-P(1)"),
    }.get(modulus)
    return (
        modulus
        if expected
        and (item["residue_basis"], item["residue_coefficients"], item["conclusion"])
        == expected
        else None
    )


def _result_is_valid(result, frozen):
    required = {
        "orbit_basis",
        "exact_period_coefficients",
        "orbit_divisibility",
        "modular_reductions",
        "infinite_prime_step",
        "polynomial_identity_step",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    reductions = result["modular_reductions"]
    parsed = (
        [_reduction(item) for item in reductions]
        if isinstance(reductions, list)
        else []
    )
    return bool(
        frozen.get("orbit_basis") == ["F(pq)", "F(p)", "F(q)", "F(1)"]
        and result["orbit_basis"] == frozen["orbit_basis"]
        and result["exact_period_coefficients"] == [1, -1, -1, 1]
        and result["orbit_divisibility"] == "pq_DIVIDES_F(pq)-F(p)-F(q)+F(1)"
        and len(parsed) == 2
        and set(parsed) == {"p", "q"}
        and result["infinite_prime_step"]
        == "FOR_EACH_PRIME_q_ALL_OTHER_PRIMES_p_DIVIDE_P(q)-P(1)_SO_P(q)=P(1)"
        and result["polynomial_identity_step"]
        == "P_EQUALS_CONSTANT_P(1)_ON_INFINITELY_MANY_PRIMES"
    )


def main():
    submission, frozen = load_submission(), _load_frozen_input()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _result_is_valid(submission.get("result"), frozen))
    evidence = None
    if (
        contract
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
    ):
        evidence = read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/periodic-orbit-certificate.json",
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
        and submission.get("scope")
        == "ALL_FUNCTIONS_Z_TO_Z_WITH_FINITE_ITERATE_FIXED_POINT_COUNTS_AND_ALL_P_IN_Z[x]"
        and submission.get("limitations")
        == ["STANDARD_ORBIT_DIVISIBILITY_AND_INFINITE_PRIME_LEMMAS_TRUSTED"]
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
