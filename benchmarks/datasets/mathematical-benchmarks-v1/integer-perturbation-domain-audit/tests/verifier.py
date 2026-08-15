import json
import re
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    aggregate_reward,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")
ALLOWED_ASSURANCES = frozenset({"COMPUTED"})
SCHEMA_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED", "VERIFIED"})


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
        "sum_values",
        "b_min",
        "b_max",
        "cancellation_indices",
    }
    if not isinstance(witness, dict) or set(witness) != required:
        return False
    contract = source.get("witness_contract", {})
    period = witness.get("period")
    if type(period) is not int or not contract.get(
        "period_min", 1
    ) <= period <= contract.get("period_max", 0):
        return False
    a_values, b_values, sums = (
        witness.get("a_values"),
        witness.get("b_values"),
        witness.get("sum_values"),
    )
    if not all(
        isinstance(values, list)
        and len(values) == period
        and all(type(x) is int for x in values)
        for values in (a_values, b_values, sums)
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
    cancel_indices = witness.get("cancellation_indices")
    if not (
        isinstance(cancel_indices, list) and all(type(x) is int for x in cancel_indices)
    ):
        return False
    return bool(
        sums == expected_sums
        and _is_exact_int(witness.get("b_min"))
        and witness.get("b_min") == min(b_values)
        and _is_exact_int(witness.get("b_max"))
        and witness.get("b_max") == max(b_values)
        and len(cancel_indices) == len(set(cancel_indices))
        and set(cancel_indices) == set(cancellations)
        and len(cancellations) >= contract.get("minimum_cancellations", period + 1)
    )


def _has_affirmative_prohibited_claim(text):
    clauses = re.split(r"[.;]|\s*,\s*(?:and|but)\s+|\s+(?:and|but)\s+", text)
    for clause in clauses:
        if not any(topic in clause for topic in ("lean", "irrational", "theorem")):
            continue
        for match in re.finditer(
            r"\b(?:verified|proved|proven|confirmed|compile|compiles|compiled|"
            r"asserted|asserts|true|valid|correct|holds|follows|established|"
            r"establishes|demonstrated|demonstrates|shown|shows)\b",
            clause,
        ):
            prefix = clause[: match.start()][-80:]
            if not re.search(r"\b(?:no|never)\b|\bnot\b(?!\s+only\b)", prefix):
                return True
    return False


def _audit_prose_valid(text):
    """Accept concise semantic paraphrases of the required audit."""
    has_natural_domain = bool(re.search(r"\b(?:natural|nat)(?:[- ]domain)?\b", text))
    has_integer_domain = bool(
        re.search(r"\binteger\b|\bz[- ]valued\b|\bz[- ]value\b", text)
    )
    return has_natural_domain and has_integer_domain


def _raw_submission():
    """Parse the bounded submission without applying the public schema."""
    path = W / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main():
    raw = _raw_submission()
    source = _frozen_source()
    input_binding = workspace_input_is_bound(W / "input.json", tests=E)
    submission = load_submission(require_input_binding=False)
    result = raw.get("result") if isinstance(raw, dict) else None
    math_ok = bool(_valid(result, source))
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=True,
        protocol_ok=bool(input_binding and submission is not None),
    )
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
