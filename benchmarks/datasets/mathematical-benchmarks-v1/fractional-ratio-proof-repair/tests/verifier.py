import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
MISMATCHES = {"OBJECTIVE_REPLACED", "BINARY_DOMAIN_RELAXED", "UNDECLARED_BUDGET_ADDED"}
LIMITATION = "The verifier certifies only the frozen exact instance; it does not machine-prove a general greedy theorem."

# Semantic clause obligations for the evidence explanation.  The public
# instruction requires explaining the three contract mismatches and why the
# residual certificate repairs the frozen objective.  Each clause is a pair
# (required_terms, alternative_terms): every required term must appear, and at
# least one alternative term must also appear.  This accepts equivalent
# phrasing while rejecting unrelated text.
_EVIDENCE_CLAUSES = (
    ({"objective", "ratio"}, {"replace", "substitut", "instead", "swap", "relax"}),
    ({"binary"}, {"domain", "relax", "fractional", "continuous", "integer"}),
    ({"budget"}, {"add", "undeclared", "unannounced", "introduce", "constraint"}),
    ({"residual"}, {"repair", "certificate", "fix", "correct", "restore", "chosen"}),
    ({"zero"}, {"maximum", "sum", "transformed"}),
)
# Read at most this many bytes for the prose semantic check.  This is not an
# evidence validity cap — the file itself has no byte limit — but a parsing
# bound so a huge valid artifact cannot exhaust verifier memory before
# reward.json is written.
_PROSE_READ_BYTES = 1_048_576


def _has_term(words: set[str], term: str) -> bool:
    """Check if any word starts with the given term stem."""

    return any(word.startswith(term) for word in words)


def _evidence_explains_clauses(text: str) -> bool:
    """Check that prose addresses every documented semantic obligation."""

    lines = [line for line in text.splitlines() if not line.startswith("RESULT_JSON:")]
    prose = " ".join(lines).lower()
    words = set(re.findall(r"[a-z]+", prose))
    if len(words) < 10:
        return False
    for required, alternatives in _EVIDENCE_CLAUSES:
        if not all(_has_term(words, term) for term in required):
            return False
        if alternatives and not any(_has_term(words, alt) for alt in alternatives):
            return False
    return True


def evidence_ok(evidence):
    # The typed residual certificate is replayed independently.  The public
    # evidence contract requires one digest-bound text artifact only.
    if not evidence_list_is_bound(evidence):
        return False
    path = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        with path.open("r", encoding="utf-8") as stream:
            head = stream.read(_PROSE_READ_BYTES)
    except (OSError, UnicodeError, RecursionError, MemoryError):
        return False
    return _evidence_explains_clauses(head)


def valid_result(result, data):
    if not isinstance(result, dict) or set(result) != {
        "contract_mismatches",
        "selected_indices",
        "attained_ratio",
        "constant_residual",
        "item_residuals",
        "positive_residual_indices",
        "maximum_residual_sum",
        "repair_method",
    }:
        return False
    mismatches = result.get("contract_mismatches")
    if (
        not isinstance(mismatches, list)
        or len(mismatches) != 3
        or not all(type(m) is str for m in mismatches)
        or set(mismatches) != MISMATCHES
    ):
        return False
    if (
        result.get("repair_method") != "EXACT_FRACTIONAL_RESIDUAL_CERTIFICATE"
        or type(result.get("maximum_residual_sum")) is not int
        or result.get("maximum_residual_sum") != 0
    ):
        return False
    selected = result.get("selected_indices")
    if (
        not isinstance(selected, list)
        or any(type(i) is not int for i in selected)
        or len(selected) != len(set(selected))
        or any(i < 0 or i >= len(data["items"]) for i in selected)
    ):
        return False
    ratio_text = result.get("attained_ratio")
    if (
        not isinstance(ratio_text, str)
        or len(ratio_text) > 64
        or re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", ratio_text) is None
    ):
        return False
    try:
        ratio = Fraction(ratio_text)
    except (ValueError, ZeroDivisionError):
        return False
    if str(ratio) != ratio_text:
        return False
    numerator = data["alpha"] + sum(data["items"][i]["t"] for i in selected)
    denominator = data["beta"] + sum(data["items"][i]["f"] for i in selected)
    if Fraction(numerator, denominator) != ratio:
        return False
    p, q = ratio.numerator, ratio.denominator
    constant = q * data["alpha"] - p * data["beta"]
    residuals = [q * item["t"] - p * item["f"] for item in data["items"]]
    submitted = result.get("item_residuals")
    if not isinstance(submitted, list) or len(submitted) != len(residuals):
        return False
    if any(
        not isinstance(row, dict)
        or set(row) != {"index", "value"}
        or type(row["index"]) is not int
        or type(row["value"]) is not int
        for row in submitted
    ):
        return False
    submitted_by_index = {row["index"]: row["value"] for row in submitted}
    if len(submitted_by_index) != len(residuals) or submitted_by_index != dict(
        enumerate(residuals)
    ):
        return False
    positives = [i for i, value in enumerate(residuals) if value > 0]
    submitted_positives = result.get("positive_residual_indices")
    return (
        type(result.get("constant_residual")) is int
        and result.get("constant_residual") == constant
        and isinstance(submitted_positives, list)
        and all(type(i) is int for i in submitted_positives)
        and len(submitted_positives) == len(set(submitted_positives))
        and set(submitted_positives) == set(positives)
        and set(selected) == set(positives)
        and constant + sum(max(0, value) for value in residuals) == 0
    )


def raw_submission():
    path = W / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def _array_preflight(raw):
    """Reject oversized index arrays before expensive schema validation."""

    if not isinstance(raw, dict):
        return True
    result = raw.get("result")
    if not isinstance(result, dict):
        return True
    for key in ("selected_indices", "positive_residual_indices", "item_residuals"):
        value = result.get(key)
        if isinstance(value, list) and len(value) > 24:
            return False
    return True


def main():
    raw = raw_submission()
    expected = json.loads((T / "expected.json").read_text())
    data = json.loads((T / "input.json").read_text())
    input_binding = workspace_input_is_bound()
    submission = (
        load_submission(W / "submission.json", require_input_binding=False)
        if _array_preflight(raw)
        else None
    )
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = raw.get("result") if isinstance(raw, dict) else None
    math_ok = valid_result(result, data)
    ev_ok = bool(isinstance(raw, dict) and evidence_ok(raw.get("evidence")))
    scope_ok = bool(
        isinstance(raw, dict)
        and raw.get("scope") == "frozen-24-item-binary-fractional-ratio-instance"
        and raw.get("completeness") == "COMPLETE"
        and raw.get("limitations") == [LIMITATION]
    )
    assurance_ok = bool(
        isinstance(raw, dict) and raw.get("claimed_assurance") == "COMPUTED"
    )
    false_cert = false_verified_claim(raw, verification_record_bound=False)
    correct = bool(
        contract and input_binding and math_ok and ev_ok and scope_ok and not false_cert
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
                "correctness": float(math_ok),
                "evidence_validity": float(ev_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": 0 if not correct else 0.9 + 0.1 * assurance_ok,
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
