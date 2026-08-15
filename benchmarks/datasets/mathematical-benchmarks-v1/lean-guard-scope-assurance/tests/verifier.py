import json
from pathlib import Path
from typing import Any

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")


def _source() -> dict[str, Any]:
    try:
        visible = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if any(
            path.is_symlink() or not path.is_file() or path.stat().st_size > 1_048_576
            for path in (visible, frozen)
        ):
            return {}
        raw = frozen.read_bytes()
        if visible.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, RecursionError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _derive(case: dict[str, Any]) -> tuple[list[str], str] | None:
    case_id = case.get("id")
    if case_id == "guard_after_binder":
        if case.get("guard") == "b_ne_zero" and case.get("result_type") == "Nat":
            return [
                "INTEGER_DIVISION_TRUNCATION"
            ], "FULL_SCOPE_GUARD_BUT_NAT_DIVISION_MAY_TRUNCATE"
    elif case_id == "allzero_div_one":
        if case.get("ofnat_zero_equals_one") is True:
            return [
                "DIVISION_BY_ZERO"
            ], "LITERAL_ONE_IS_DEFINITIONALLY_ZERO_IN_CUSTOM_TYPE"
    elif case_id == "allzero_bad_lt_guard":
        if (
            case.get("ofnat_zero_equals_one") is True
            and case.get("lt_is_universal") is True
        ):
            return ["DIVISION_BY_ZERO"], "ARBITRARY_LT_DOES_NOT_ESTABLISH_NONZERO"
    elif case_id == "proof_term_noise":
        if (
            case.get("expression_location") == "proof_term"
            and case.get("declaration_kind") == "theorem"
        ):
            return [], "PROOF_TERMS_EXCLUDED_FROM_STATEMENT_ANALYSIS"
    elif (
        case_id == "zero_over_variable"
        and case.get("numerator") == "zero"
        and case.get("guard") is None
    ):
        return ["DIVISION_BY_ZERO"], "ZERO_NUMERATOR_IS_EXACT_BUT_DIVISOR_IS_UNGUARDED"
    return None


def _result_item_ok(item, case):
    if (
        not isinstance(item, dict)
        or set(item) != {"id", "findings", "reason"}
        or not isinstance(case, dict)
    ):
        return False
    derived = _derive(case)
    if derived is None:
        return False
    findings, reason = derived
    return not (
        item.get("id") != case.get("id")
        or item.get("findings") != findings
        or item.get("reason") != reason
    )


def _result(value: object, source: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or set(value) != {"cases"}:
        return False
    submitted = value.get("cases")
    cases = source.get("cases")
    provenance = source.get("source", {})
    if not isinstance(submitted, list) or not isinstance(cases, list):
        return False
    if len(submitted) != len(cases) or len(cases) != 5:
        return False
    if provenance.get("revision") != "3e7e99d027fece04d9cd96288cdd040c366458e5":
        return False
    for item, case in zip(submitted, cases, strict=True):
        if not _result_item_ok(item, case):
            return False
    return True


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    correct = bool(_result(data.get("result"), _source()))
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correct),
                "reward": float(correct),
            },
            sort_keys=True,
        )
        + "\n"
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
