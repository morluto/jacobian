import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")

# Lemma forms that, when used affirmatively about the open problem, claim it is
# solved or machine/formally verified. Base present-tense verbs ("solve",
# "verify") are intentionally excluded so legitimate prose such as "the verifier
# recomputes" or "does not solve" is not flagged.
_AFFIRMATIVE_LEMMAS = (
    "solved",
    "solves",
    "verified",
    "proved",
    "proven",
    "proves",
    "certified",
    "settled",
)
_NEGATION_TOKENS = (
    "not",
    "no",
    "without",
    "doesnt",
    "cannot",
    "cant",
    "never",
    "nor",
    "neither",
)
_SCOPE_LIMITING_MARKERS = (
    "not",
    "no claim",
    "no ",
    "does not",
    "doesn't",
    "doesnt",
    "without",
    "cannot",
    "can't",
    "never",
    "nor",
    "neither",
    "unsolved",
    "not solved",
    "not verified",
    "not machine",
)


def _fraction(text):
    if not isinstance(text, str):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if str(value) == text else None


def _words(text):
    return re.findall(r"[a-z]+", text.casefold())


def _affirmative_solved_or_verified_claim(text):
    """True if text affirmatively claims the open problem is solved or verified.

    A claim is affirmative when a solved/verified lemma appears without a
    negation token in the preceding clause, so negated scope statements such as
    "does not solve" or "not machine verified" are accepted.
    """
    if re.search(r"\b(machine|formally?)\s+(verified|certified)\b", text.casefold()):
        return True
    for clause in re.split(r"[.!?;]+", text.casefold()):
        if not any(subject in clause for subject in ("open problem", "theorem", "lean")):
            continue
        words = _words(clause)
        for index, word in enumerate(words):
            if word not in _AFFIRMATIVE_LEMMAS:
                continue
            if not any(token in words[max(0, index - 2) : index] for token in _NEGATION_TOKENS):
                return True
    return False


def _limitation_is_scope_limiting(text):
    if not isinstance(text, str):
        return False
    folded = text.casefold()
    if "open problem" not in folded:
        return False
    if _affirmative_solved_or_verified_claim(text):
        return False
    return any(marker in folded for marker in _SCOPE_LIMITING_MARKERS)


def _model(value, *, intended, proposed, bound, lower, upper):
    if not isinstance(value, dict) or set(value) != {
        "limsup_values",
        "intended_truth",
        "proposed_truth",
        "distinguishing_index",
    }:
        return False
    raw = value["limsup_values"]
    if not isinstance(raw, list) or not 2 <= len(raw) <= 8:
        return False
    values = [_fraction(item) for item in raw]
    if any(item is None or item < lower or item > upper for item in values):
        return False
    intended_truth = any(item <= bound for item in values)
    proposed_truth = all(item >= bound for item in values)
    index = value["distinguishing_index"]
    if type(index) is not int or not 0 <= index < len(values):
        return False
    distinguishing = values[index]
    index_valid = distinguishing <= bound if intended else distinguishing < bound
    if proposed and not intended:
        index_valid = distinguishing > bound
    return bool(
        value["intended_truth"] is intended_truth is intended
        and value["proposed_truth"] is proposed_truth is proposed
        and index_valid
    )


def _valid_result(result, source):
    if not isinstance(result, dict) or set(result) != {
        "relationship",
        "intended_only_model",
        "proposed_only_model",
    }:
        return False
    bound = Fraction(source["bound"])
    limits = source["model_constraints"]
    lower, upper = Fraction(limits["minimum_value"]), Fraction(limits["maximum_value"])
    return bool(
        result["relationship"] == "INCOMPARABLE"
        and _model(
            result["intended_only_model"],
            intended=True,
            proposed=False,
            bound=bound,
            lower=lower,
            upper=upper,
        )
        and _model(
            result["proposed_only_model"],
            intended=False,
            proposed=True,
            bound=bound,
            lower=lower,
            upper=upper,
        )
    )


def _evidence_valid(evidence, result):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
    except (OSError, UnicodeError):
        return False
    if _affirmative_solved_or_verified_claim(text):
        return False
    markers = [
        line.removeprefix("RESULT_JSON:").strip()
        for line in text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    if len(markers) != 1:
        return False
    try:
        bound_result = json.loads(markers[0])
    except ValueError:
        return False
    if not isinstance(result, dict) or bound_result != result:
        return False
    folded = text.casefold()
    return (
        all(word in folded for word in ("existential", "universal", "incomparable"))
        and "not incomparable" not in folded
    )


def _evidence_affirmative_claim(evidence):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        return _affirmative_solved_or_verified_claim(target.read_text())
    except (OSError, UnicodeError):
        return False


def main():
    submission = load_submission()
    source = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result")
    math_correct = bool(contract and _valid_result(result, source))
    evidence_valid = bool(
        contract and math_correct and _evidence_valid(data.get("evidence"), result)
    )
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = data.get("limitations", [])
    limitations_affirmative = any(
        _affirmative_solved_or_verified_claim(item)
        for item in limitations
        if isinstance(item, str)
    )
    limitations_correct = bool(
        contract
        and not limitations_affirmative
        and any(
            _limitation_is_scope_limiting(item)
            for item in limitations
            if isinstance(item, str)
        )
    )
    false_certification = bool(
        false_verified_claim(submission, verification_record_bound=False)
        or limitations_affirmative
        or _evidence_affirmative_claim(data.get("evidence"))
        or data.get("claimed_assurance") not in {"UNVERIFIED", "COMPUTED"}
    )
    correct = bool(
        contract and math_correct and limitations_correct and not false_certification
    )
    reward = (
        0
        if not correct
        else 0.7 + 0.1 * evidence_valid + 0.1 * scope_correct + 0.1 * assurance_correct
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
