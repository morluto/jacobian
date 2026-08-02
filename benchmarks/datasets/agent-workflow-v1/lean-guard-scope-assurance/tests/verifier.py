import json
import re
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
LIMITATION = "The verifier derives semantic classifications from frozen facts but does not run Lean or certify the upstream linter implementation."


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
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "findings", "reason"}
            or not isinstance(case, dict)
        ):
            return False
        derived = _derive(case)
        if derived is None:
            return False
        findings, _reason = derived
        submitted_reason = item.get("reason")
        if not isinstance(submitted_reason, str) or len(submitted_reason.strip()) < 12:
            return False
        reason_text = submitted_reason.casefold()
        if (
            findings
            and "DIVISION_BY_ZERO" in findings
            and (
                not ("zero" in reason_text or "divisor" in reason_text)
                or re.search(
                    r"(?:division|divisor|zero)[^.;\n]{0,60}"
                    r"(?:impossible|not possible|cannot|can['']?t|safe|never)\b",
                    reason_text,
                )
                or re.search(
                    r"\b(?:no|not)\b[^.;\n]{0,40}\b(?:division|divisor)\b",
                    reason_text,
                )
            )
        ):
            return False
        if (
            findings
            and "INTEGER_DIVISION_TRUNCATION" in findings
            and (
                "truncat" not in reason_text
                or re.search(
                    r"\b(?:no|not|never|doesn['']?t|cannot)\b[^.;\n]{0,30}truncat",
                    reason_text,
                )
            )
        ):
            return False
        if not findings and not any(
            term in reason_text for term in ("proof", "statement", "term")
        ):
            return False
        if item.get("id") != case.get("id") or item.get("findings") != findings:
            return False
    return True


def _evidence(value: object, result: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not evidence_list_is_bound(value)
    ):
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        if path.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        raw_text = path.read_text()
    except (OSError, UnicodeError):
        return False
    markers = [
        line.removeprefix("RESULT_JSON:").strip()
        for line in raw_text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    if len(markers) != 1:
        return False
    try:
        if json.loads(markers[0]) != result:
            return False
    except (ValueError, RecursionError):
        return False
    text = raw_text.casefold()
    return all(
        term in text
        for term in ("full proof-state", "allzero", "proof-term", "computed")
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    correct = bool(contract and _result(data.get("result"), _source()))
    evidence = bool(correct and _evidence(data.get("evidence"), data.get("result")))
    scope = bool(contract and data.get("scope") == expected["required_scope"])
    assurance = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = bool(contract and data.get("limitations") == [LIMITATION])
    false_verified = false_verified_claim(submission, verification_record_bound=False)
    passed = bool(
        correct
        and evidence
        and scope
        and assurance
        and limitations
        and not false_verified
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correct),
                "evidence_validity": float(evidence),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": float(passed),
                "false_certification": false_verified,
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
