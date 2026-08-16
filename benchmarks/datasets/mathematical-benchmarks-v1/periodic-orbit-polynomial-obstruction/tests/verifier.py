import json
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W, E = (Path("/app"), Path("/tests"))
MAX_SUBMISSION_BYTES = 1048576
MAX_INPUT_BYTES = 1048576
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


def _reduction(item):
    if not isinstance(item, dict) or set(item) != {
        "modulus",
        "residue_basis",
        "residue_coefficients",
        "conclusion",
    }:
        return None
    modulus = item["modulus"]
    expected_basis = {"p": ["P(q)", "P(1)"], "q": ["P(p)", "P(1)"]}.get(modulus)
    expected_conclusion = {"p": "p_DIVIDES_P(q)-P(1)", "q": "q_DIVIDES_P(p)-P(1)"}.get(
        modulus
    )
    if expected_basis is None or expected_conclusion is None:
        return None
    coefficients = item["residue_coefficients"]
    if item["residue_basis"] != expected_basis:
        return None
    if item["conclusion"] != expected_conclusion:
        return None
    if not _int_list(coefficients) or len(coefficients) != 2:
        return None
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
        and (coefficients == [1, -1, -1, 1])
        and (result["orbit_divisibility"] == "pq_DIVIDES_F(pq)-F(p)-F(q)+F(1)")
        and (len(parsed) == 2)
        and (set(parsed) == {"p", "q"})
        and isinstance(infinite_step, str)
        and (infinite_step == _INFINITE_PRIME_STEP)
        and isinstance(identity_step, str)
        and (identity_step == _POLYNOMIAL_IDENTITY_STEP)
    )


def main():
    submission, frozen = (_load_submission(), _load_frozen_input())
    contract = bool(submission)
    math_correct = bool(contract and _result_is_valid(submission.get("result"), frozen))
    reward = float(math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
