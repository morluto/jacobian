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


def _limitations_valid(value: object) -> bool:
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return False
    text = value[0].casefold()
    return (
        any(
            term in text
            for term in ("frozen", "exact instance", "24-item", "binary instance")
        )
        and any(term in text for term in ("greedy theorem", "general theorem"))
        and any(
            term in text
            for term in ("not prove", "does not prove", "not machine", "only")
        )
    )


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
_EVIDENCE_STEMS = frozenset(
    term
    for required, alternatives in _EVIDENCE_CLAUSES
    for term in required | alternatives
)
_EVIDENCE_CONTRADICTIONS = (
    re.compile(r"\bratio\s+objective\b.{0,32}\b(?:unchanged|not\s+replaced)\b"),
    re.compile(r"\bbinary\s+domain\b.{0,32}\b(?:not\s+relaxed|unchanged)\b"),
    re.compile(r"\b(?:no|without)\b.{0,24}\bbudget\b.{0,24}\b(?:added|introduced)\b"),
    re.compile(r"\bresidual\s+certificate\b.{0,32}\bdoes\s+not\s+repair\b"),
    re.compile(r"\bmaximum\b.{0,32}\bresidual\b.{0,24}\bnon[- ]?zero\b"),
)


def _finish_token(token, matched_stems, sample_words):
    if not token:
        return token
    if len(sample_words) < 10:
        sample_words.add(token)
    matched_stems.update(stem for stem in _EVIDENCE_STEMS if token.startswith(stem))
    return ""


def _consume(character, token, matched_stems, sample_words):
    if "a" <= character <= "z":
        if len(token) < 32:
            token += character
        return token
    return _finish_token(token, matched_stems, sample_words)


def _clauses_satisfied(matched_stems):
    for required, alternatives in _EVIDENCE_CLAUSES:
        if not required <= matched_stems:
            return False
        if alternatives and not (alternatives & matched_stems):
            return False
    return True


def _process_newline(state):
    if not state["skip_line"]:
        if state["at_line_start"]:
            for prefix_character in state["line_prefix"]:
                state["token"] = _consume(
                    prefix_character,
                    state["token"],
                    state["matched_stems"],
                    state["sample_words"],
                )
        state["token"] = _finish_token(
            state["token"], state["matched_stems"], state["sample_words"]
        )
    state["line_prefix"] = ""
    state["at_line_start"] = True
    state["skip_line"] = False


def _process_line_start_char(character, state):
    state["line_prefix"] += character
    marker = "result_json:"
    if marker.startswith(state["line_prefix"]):
        if state["line_prefix"] == marker:
            state["skip_line"] = True
            state["at_line_start"] = False
            state["line_prefix"] = ""
        return
    for prefix_character in state["line_prefix"]:
        state["token"] = _consume(
            prefix_character,
            state["token"],
            state["matched_stems"],
            state["sample_words"],
        )
    state["line_prefix"] = ""
    state["at_line_start"] = False


def _parse_evidence_stream(path):
    matched_stems: set[str] = set()
    sample_words: set[str] = set()
    state = {
        "token": "",
        "line_prefix": "",
        "at_line_start": True,
        "skip_line": False,
        "matched_stems": matched_stems,
        "sample_words": sample_words,
    }
    contradicted = False
    carry = ""
    try:
        with path.open("r", encoding="utf-8") as stream:
            while chunk := stream.read(65_536):
                window = (carry + chunk).lower()
                contradicted = contradicted or any(
                    pattern.search(window) for pattern in _EVIDENCE_CONTRADICTIONS
                )
                carry = window[-256:]
                for character in chunk.lower():
                    if character == "\n":
                        _process_newline(state)
                        continue
                    if state["skip_line"]:
                        continue
                    if state["at_line_start"]:
                        _process_line_start_char(character, state)
                        continue
                    state["token"] = _consume(
                        character, state["token"], matched_stems, sample_words
                    )
        if not state["skip_line"]:
            if state["at_line_start"]:
                for prefix_character in state["line_prefix"]:
                    state["token"] = _consume(
                        prefix_character,
                        state["token"],
                        matched_stems,
                        sample_words,
                    )
            state["token"] = _finish_token(state["token"], matched_stems, sample_words)
    except (OSError, UnicodeError, MemoryError):
        return None
    return matched_stems, sample_words, contradicted


def _evidence_explains_clauses(path: Path) -> bool:
    """Stream prose, ignoring private result markers, with bounded parser state."""

    parsed = _parse_evidence_stream(path)
    if parsed is None:
        return False
    matched_stems, sample_words, contradicted = parsed
    if contradicted or len(sample_words) < 10:
        return False
    return _clauses_satisfied(matched_stems)


def evidence_ok(evidence):
    # The typed residual certificate is replayed independently.  The public
    # evidence contract requires one digest-bound text artifact only.
    if not evidence_list_is_bound(evidence):
        return False
    path = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    return _evidence_explains_clauses(path)


def _mismatches_ok(mismatches):
    return (
        isinstance(mismatches, list)
        and len(mismatches) == 3
        and all(type(m) is str for m in mismatches)
        and set(mismatches) == MISMATCHES
    )


def _repair_method_ok(result):
    return (
        result.get("repair_method") == "EXACT_FRACTIONAL_RESIDUAL_CERTIFICATE"
        and type(result.get("maximum_residual_sum")) is int
        and result.get("maximum_residual_sum") == 0
    )


def _selected_indices_ok(selected, data):
    return (
        isinstance(selected, list)
        and all(type(i) is int for i in selected)
        and len(selected) == len(set(selected))
        and all(0 <= i < len(data["items"]) for i in selected)
    )


def _parse_ratio(ratio_text):
    if (
        not isinstance(ratio_text, str)
        or len(ratio_text) > 64
        or re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", ratio_text) is None
    ):
        return None
    try:
        ratio = Fraction(ratio_text)
    except (ValueError, ZeroDivisionError):
        return None
    return ratio if str(ratio) == ratio_text else None


def _residuals_ok(submitted, residuals):
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
    return len(submitted_by_index) == len(residuals) and submitted_by_index == dict(
        enumerate(residuals)
    )


def _positives_ok(result, positives, selected, constant, residuals):
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
    if not _mismatches_ok(result.get("contract_mismatches")):
        return False
    if not _repair_method_ok(result):
        return False
    selected = result.get("selected_indices")
    if not _selected_indices_ok(selected, data):
        return False
    ratio = _parse_ratio(result.get("attained_ratio"))
    if ratio is None:
        return False
    numerator = data["alpha"] + sum(data["items"][i]["t"] for i in selected)
    denominator = data["beta"] + sum(data["items"][i]["f"] for i in selected)
    if Fraction(numerator, denominator) != ratio:
        return False
    p, q = ratio.numerator, ratio.denominator
    constant = q * data["alpha"] - p * data["beta"]
    residuals = [q * item["t"] - p * item["f"] for item in data["items"]]
    if not _residuals_ok(result.get("item_residuals"), residuals):
        return False
    positives = [i for i, value in enumerate(residuals) if value > 0]
    return _positives_ok(result, positives, selected, constant, residuals)


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
        and _limitations_valid(raw.get("limitations"))
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
                "reward": float(correct),
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
