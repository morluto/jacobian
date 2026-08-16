import json
from pathlib import Path

from verifier_support import (
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _frozen_source():
    """Load the trusted frozen input from /tests without reading workspace bytes."""
    try:
        frozen = E / "input.json"
        if frozen.is_symlink() or not is_regular_bounded_file(frozen, max_bytes=None):
            return {}
        value = json.loads(frozen.read_bytes())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_exact_int(value):
    """Reject JSON booleans that compare equal to 0 or 1."""
    return type(value) is int


def _nat_redundancy_valid(redundancy):
    """Validate the symbolic Nat redundancy certificate with exact int types."""
    if not isinstance(redundancy, dict):
        return False
    if set(redundancy) != {
        "a_lower_bound",
        "b_lower_bound",
        "sum_lower_bound",
        "rule",
    }:
        return False
    return bool(
        _is_exact_int(redundancy.get("a_lower_bound"))
        and redundancy.get("a_lower_bound") == 0
        and _is_exact_int(redundancy.get("b_lower_bound"))
        and redundancy.get("b_lower_bound") == 1
        and _is_exact_int(redundancy.get("sum_lower_bound"))
        and redundancy.get("sum_lower_bound") == 1
        and redundancy.get("rule") == "ORDERED_ADDITION_LOWER_BOUND"
    )


def _valid(result, source):
    if not isinstance(result, dict) or set(result) != {
        "semantic_status",
        "nat_redundancy",
        "integer_witness",
    }:
        return False
    if result.get("semantic_status") != "STRICTLY_WEAKER":
        return False
    if not _nat_redundancy_valid(result.get("nat_redundancy")):
        return False
    witness = result.get("integer_witness")
    required = {
        "period",
        "a_values",
        "b_values",
    }
    if not isinstance(witness, dict) or set(witness) != required:
        return False
    contract = source.get("witness_contract", {})
    period = witness.get("period")
    if type(period) is not int or not contract.get(
        "period_min", 1
    ) <= period <= contract.get("period_max", 0):
        return False
    a_values, b_values = (
        witness.get("a_values"),
        witness.get("b_values"),
    )
    if not all(
        isinstance(values, list)
        and len(values) == period
        and all(type(x) is int for x in values)
        for values in (a_values, b_values)
    ):
        return False
    limit = contract.get("value_abs_max", 0)
    if any(not 1 <= x <= limit for x in a_values) or any(
        x == 0 or abs(x) > limit for x in b_values
    ):
        return False
    if not (any(x < 0 for x in b_values) and any(x > 0 for x in b_values)):
        return False
    expected_sums = [a + b for a, b in zip(a_values, b_values, strict=True)]
    cancellations = [i for i, value in enumerate(expected_sums) if value == 0]
    return bool(len(cancellations) >= contract.get("minimum_cancellations", period + 1))


def main():
    source = _frozen_source()
    input_binding = workspace_input_is_bound(W / "input.json", tests=E)
    submission = load_submission(require_input_binding=False)
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = bool(_valid(result, source))
    reward = float(math_ok and input_binding and submission is not None)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
