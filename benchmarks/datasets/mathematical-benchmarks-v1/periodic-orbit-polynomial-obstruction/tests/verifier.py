import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")
MAX_SUBMISSION_BYTES = 1_048_576
MAX_INPUT_BYTES = 1_048_576
MAX_EVIDENCE_BYTES = 1_048_576

_ASSURANCE_ORDER = {
    "UNVERIFIED": 0,
    "COMPUTED": 1,
    "CHECKED": 2,
    "VERIFIED": 3,
}

# Acceptable proof-step identifiers.  The schema exposes a finite enum so the
# agent knows the expected format without seeing the exact answer as a single
# const; the verifier accepts only the correct one.
_INFINITE_PRIME_STEP = (
    "FOR_EACH_PRIME_q_ALL_OTHER_PRIMES_p_DIVIDE_P(q)-P(1)_SO_P(q)=P(1)"
)
_POLYNOMIAL_IDENTITY_STEP = "P_EQUALS_CONSTANT_P(1)_ON_INFINITELY_MANY_PRIMES"


def _load_submission():
    """Bound submission size before reading to avoid OOM on oversized files."""
    path = W / "submission.json"
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_SUBMISSION_BYTES
        ):
            return None
    except OSError:
        return None
    return load_submission(path)


def _load_frozen_input():
    """Load the frozen input, bounding the workspace copy before reading."""
    try:
        frozen_path = E / "input.json"
        workspace_path = W / "input.json"
        if frozen_path.is_symlink() or workspace_path.is_symlink():
            return {}
        if (
            not workspace_path.is_file()
            or workspace_path.stat().st_size > MAX_INPUT_BYTES
        ):
            return {}
        raw = frozen_path.read_bytes()
        if workspace_path.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_int(value):
    """Reject booleans and floats; accept only exact Python integers."""
    return type(value) is int


def _int_list(value):
    """Validate a list of exact integers, rejecting booleans and floats."""
    return isinstance(value, list) and all(_is_int(item) for item in value)


def _json_equal(left, right):
    """Compare two JSON values without Python's bool/int coercion."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _read_bounded_evidence(descriptor, *, expected_path):
    """Resolve and parse evidence, rejecting oversized files before parsing."""
    target = resolve_evidence(
        descriptor,
        expected_path=expected_path,
        workspace=W,
    )
    if target is None:
        return None
    try:
        if target.stat().st_size > MAX_EVIDENCE_BYTES:
            return None
        value = json.loads(target.read_text())
    except (OSError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _reduction(item):
    if not isinstance(item, dict) or set(item) != {
        "modulus",
        "residue_basis",
        "residue_coefficients",
        "conclusion",
    }:
        return None
    modulus = item["modulus"]
    expected_basis = {
        "p": ["P(q)", "P(1)"],
        "q": ["P(p)", "P(1)"],
    }.get(modulus)
    expected_conclusion = {
        "p": "p_DIVIDES_P(q)-P(1)",
        "q": "q_DIVIDES_P(p)-P(1)",
    }.get(modulus)
    if expected_basis is None or expected_conclusion is None:
        return None
    coefficients = item["residue_coefficients"]
    if item["residue_basis"] != expected_basis:
        return None
    if item["conclusion"] != expected_conclusion:
        return None
    if not _int_list(coefficients) or len(coefficients) != 2:
        return None
    # Accept sign-equivalent coefficient vectors: divisibility is invariant
    # under multiplication by -1, so [-1, 1] and [1, -1] are both valid.
    if coefficients not in ([-1, 1], [1, -1]):
        return None
    return modulus


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
    coefficients = result["exact_period_coefficients"]
    if not _int_list(coefficients) or len(coefficients) != 4:
        return False
    reductions = result["modular_reductions"]
    parsed = (
        [_reduction(item) for item in reductions]
        if isinstance(reductions, list)
        else []
    )
    infinite_step = result["infinite_prime_step"]
    identity_step = result["polynomial_identity_step"]
    return bool(
        frozen.get("orbit_basis") == ["F(pq)", "F(p)", "F(q)", "F(1)"]
        and result["orbit_basis"] == frozen["orbit_basis"]
        and coefficients == [1, -1, -1, 1]
        and result["orbit_divisibility"] == "pq_DIVIDES_F(pq)-F(p)-F(q)+F(1)"
        and len(parsed) == 2
        and set(parsed) == {"p", "q"}
        and isinstance(infinite_step, str)
        and infinite_step == _INFINITE_PRIME_STEP
        and isinstance(identity_step, str)
        and identity_step == _POLYNOMIAL_IDENTITY_STEP
    )


def main():
    submission, frozen = _load_submission(), _load_frozen_input()
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
        evidence = _read_bounded_evidence(
            submission["evidence"][0],
            expected_path="evidence/periodic-orbit-certificate.json",
        )
    evidence_valid = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == expected["task_id"]
        and _json_equal(evidence["result"], submission.get("result"))
        and _json_equal(evidence["limitations"], submission.get("limitations"))
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
    claimed = (
        submission.get("claimed_assurance") if isinstance(submission, dict) else None
    )
    ceiling = expected["maximum_assurance"]
    assurance_within_ceiling = bool(
        contract
        and isinstance(claimed, str)
        and isinstance(ceiling, str)
        and claimed in _ASSURANCE_ORDER
        and ceiling in _ASSURANCE_ORDER
        and _ASSURANCE_ORDER[claimed] <= _ASSURANCE_ORDER[ceiling]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    reward = aggregate_reward(
        correctness=math_correct,
        evidence_validity=evidence_valid,
        scope_accuracy=envelope,
        assurance_calibration=assurance,
        false_certification=false_certification,
        protocol_ok=bool(contract and assurance_within_ceiling),
        soft_assurance=True,
    )
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
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
