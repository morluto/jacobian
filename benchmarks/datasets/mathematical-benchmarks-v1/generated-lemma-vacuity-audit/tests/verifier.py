import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
LIMITATION = (
    "Lean parsing, elaboration, compilation, and the truth of the original IMO "
    "theorems are not assessed."
)
CORRECTED_CONTRACT = {
    "quantifier": "FORALL",
    "variable": "k",
    "body": {
        "operator": "IMPLIES",
        "antecedent": "a_plus_b_equals_k_squared",
        "consequent": "k_between_declared_bounds",
    },
}


def _load_frozen_input() -> dict:
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


def _exact_int(value: object) -> bool:
    return type(value) is int


def _square_witness(
    certificate: object,
    source: dict,
    *,
    require_antecedent: bool,
) -> bool:
    fields = {
        "a",
        "b",
        "k",
        "antecedent_holds",
        "frozen_formula_holds",
        "bounds_hold",
    }
    if not isinstance(certificate, dict) or set(certificate) != fields:
        return False
    if not all(_exact_int(certificate[name]) for name in ("a", "b", "k")):
        return False
    case = source.get("square_bound_case")
    if not isinstance(case, dict):
        return False
    interval = case.get("card_interval")
    bounds = case.get("declared_bounds")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(_exact_int(value) for value in interval)
        or not isinstance(bounds, dict)
        or not all(_exact_int(bounds.get(name)) for name in ("k_min", "k_max"))
    ):
        return False
    a, b, k = certificate["a"], certificate["b"], certificate["k"]
    in_scope = interval[0] <= a <= interval[1] and interval[0] <= b <= interval[1]
    antecedent = a + b == k * k
    within_bounds = bounds["k_min"] <= k <= bounds["k_max"]
    frozen_formula = (not antecedent) or within_bounds
    claimed_flags_match = (
        certificate["antecedent_holds"] is antecedent
        and certificate["bounds_hold"] is within_bounds
        and certificate["frozen_formula_holds"] is frozen_formula
    )
    if not (in_scope and a != b and k >= 0 and claimed_flags_match):
        return False
    if require_antecedent:
        return antecedent and within_bounds and frozen_formula
    return not antecedent and not within_bounds and frozen_formula


def _square_audit_is_valid(audit: object, source: dict) -> bool:
    fields = {
        "defect",
        "vacuity_witness",
        "corrected_contract_ast",
        "nonvacuous_check",
    }
    return bool(
        isinstance(audit, dict)
        and set(audit) == fields
        and audit["defect"] == "EXISTENTIAL_IMPLICATION_VACUITY"
        and audit["corrected_contract_ast"] == CORRECTED_CONTRACT
        and _square_witness(audit["vacuity_witness"], source, require_antecedent=False)
        and _square_witness(audit["nonvacuous_check"], source, require_antecedent=True)
    )


def _divides(divisor: int, dividend: int) -> bool:
    return divisor != 0 and dividend % divisor == 0


def _divisor_audit_is_valid(audit: object) -> bool:
    fields = {
        "defect",
        "a",
        "b",
        "d",
        "dividends",
        "original_premise_holds",
    }
    if not isinstance(audit, dict) or set(audit) != fields:
        return False
    if not all(_exact_int(audit[name]) for name in ("a", "b", "d")):
        return False
    a, b, d = audit["a"], audit["b"], audit["d"]
    dividends = audit["dividends"]
    if (
        audit["defect"] != "TRIVIAL_EXISTENTIAL_WITNESS"
        or not 1 <= a <= 1_000_000
        or not 1 <= b <= 1_000_000
        or not 1 <= d <= 1_000_000
        or a == b
        or d != 1
        or not isinstance(dividends, list)
        or len(dividends) != 3
        or not all(_exact_int(value) for value in dividends)
    ):
        return False
    actual_dividends = [4 * a * b - 1, 2 * a - 1, 2 * a + 1]
    original_numerator = (4 * a * a - 1) ** 2
    original_premise = _divides(actual_dividends[0], original_numerator)
    return bool(
        dividends == actual_dividends
        and all(_divides(d, value) for value in actual_dividends)
        and audit["original_premise_holds"] is original_premise
        and not original_premise
    )


def _result_is_valid(result: object, source: dict) -> bool:
    return bool(
        isinstance(result, dict)
        and set(result)
        == {"square_bound_audit", "common_divisor_audit", "source_theorem_status"}
        and result["source_theorem_status"] == "NOT_ASSESSED"
        and source.get("audit_scope", {}).get("lean_compilation") is False
        and source.get("audit_scope", {}).get("source_theorem_truth") == "NOT_ASSESSED"
        and _square_audit_is_valid(result["square_bound_audit"], source)
        and _divisor_audit_is_valid(result["common_divisor_audit"])
    )


def main() -> None:
    submission = load_submission()
    source = _load_frozen_input()
    protocol_ok = submission is not None
    result = submission.get("result") if protocol_ok else None
    math_correct = bool(protocol_ok and _result_is_valid(result, source))
    reward = float(math_correct)

    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
