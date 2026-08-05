import itertools
import json
import re
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")
ALLOWED_ASSURANCES = frozenset({"UNVERIFIED", "COMPUTED"})
REQUIRED_DEFECTS = {
    "OUTER_PARAMETER_SHADOWED",
    "WHOLE_SET_SUM_REPLACES_SUBSET_SUM",
}


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


def _subsets(values):
    return [
        tuple(combination)
        for size in range(len(values) + 1)
        for combination in itertools.combinations(values, size)
    ]


def _canonical_set(value, universe):
    return bool(
        isinstance(value, list)
        and all(type(entry) is int for entry in value)
        and value == sorted(set(value))
        and set(value) <= set(universe)
    )


def _legacy_valid(candidate, target):
    return sum(candidate) != target


def _intended_valid(candidate, target):
    return all(sum(subset) != target for subset in _subsets(candidate))


def _extremum(universe, target, predicate):
    candidates = _subsets(universe)
    return max(
        len(candidate) for candidate in candidates if predicate(candidate, target)
    )


def _shadow_extremum(multiplier, target):
    return _extremum(list(range(1, multiplier * target + 1)), target, _legacy_valid)


def _is_exact_int(value):
    """Reject JSON booleans that compare equal to 0 or 1."""
    return type(value) is int


def _shadowing_certified(value, source):
    if not isinstance(value, dict) or set(value) != {
        "target",
        "first_multiplier",
        "second_multiplier",
        "first_extremum",
        "second_extremum",
    }:
        return False
    target = source.get("shadow_instance", {}).get("target")
    allowed = source.get("shadow_instance", {}).get("allowed_cutoff_multipliers")
    first = value.get("first_multiplier")
    second = value.get("second_multiplier")
    if not all(_is_exact_int(item) for item in (first, second)):
        return False
    if (
        not _is_exact_int(value.get("target"))
        or value.get("target") != target
        or first == second
        or first not in allowed
        or second not in allowed
    ):
        return False
    first_actual = _shadow_extremum(first, target)
    second_actual = _shadow_extremum(second, target)
    return bool(
        _is_exact_int(value.get("first_extremum"))
        and _is_exact_int(value.get("second_extremum"))
        and value.get("first_extremum") == first_actual
        and value.get("second_extremum") == second_actual
        and first_actual != second_actual
    )


def _predicate_certified(value, source):
    if not isinstance(value, dict) or set(value) != {
        "target",
        "universe",
        "legacy_extremum",
        "intended_extremum",
        "legacy_witness",
        "intended_witness",
        "blocking_subset",
    }:
        return False
    instance = source.get("predicate_instance", {})
    target = instance.get("target")
    universe = instance.get("universe")
    if (
        not _is_exact_int(value.get("target"))
        or value.get("target") != target
        or not isinstance(value.get("universe"), list)
        or not all(type(entry) is int for entry in value["universe"])
        or value.get("universe") != universe
    ):
        return False
    legacy = value.get("legacy_witness")
    intended = value.get("intended_witness")
    blocker = value.get("blocking_subset")
    if not all(
        _canonical_set(candidate, universe) for candidate in (legacy, intended, blocker)
    ):
        return False
    legacy_max = _extremum(universe, target, _legacy_valid)
    intended_max = _extremum(universe, target, _intended_valid)
    return bool(
        _is_exact_int(value.get("legacy_extremum"))
        and _is_exact_int(value.get("intended_extremum"))
        and value.get("legacy_extremum") == legacy_max
        and value.get("intended_extremum") == intended_max
        and len(legacy) == legacy_max
        and _legacy_valid(legacy, target)
        and len(intended) == intended_max
        and _intended_valid(intended, target)
        and set(blocker) <= set(legacy)
        and sum(blocker) == target
        and legacy_max > intended_max
    )


def _valid_audit(result, source):
    if not isinstance(result, dict) or set(result) != {
        "semantic_status",
        "defects",
        "shadowing_certificate",
        "predicate_certificate",
    }:
        return False
    defects = result.get("defects")
    return bool(
        source.get("audit_scope", {}).get("lean_compilation") is False
        and result.get("semantic_status") == "NOT_EQUIVALENT"
        and isinstance(defects, list)
        and len(defects) == 2
        and all(type(defect) is str for defect in defects)
        and set(defects) == REQUIRED_DEFECTS
        and _shadowing_certified(result.get("shadowing_certificate"), source)
        and _predicate_certified(result.get("predicate_certificate"), source)
    )


def _limitations_valid(limitations):
    """Reject affirmative Lean/asymptotic verification claims.

    The task prohibits claiming Lean compilation or the corrected asymptotic
    conjecture was verified. A valid limitation must explicitly state that Lean
    compilation is not assessed, and must not assert that either Lean or the
    corrected asymptotic conjecture was verified.
    """
    if not isinstance(limitations, list):
        return False
    has_lean_not_assessed = False
    for item in limitations:
        if not isinstance(item, str):
            return False
        folded = item.casefold()
        if "lean" in folded and any(
            phrase in folded
            for phrase in ("not assessed", "not verified", "not proved", "not proven")
        ):
            has_lean_not_assessed = True
        if _has_affirmative_out_of_scope_claim(folded):
            return False
    return has_lean_not_assessed


def _has_affirmative_out_of_scope_claim(text):
    """Reject positive proof claims while allowing negated disclaimers."""
    words = re.findall(r"[a-z]+", text.casefold())
    proof_terms = {"verified", "proved", "proven", "confirmed"}
    subjects = {"lean", "asymptotic", "conjecture"}
    negations = {"no", "not", "never", "unverified", "unproved", "unproven"}
    for index, word in enumerate(words):
        if word not in proof_terms:
            continue
        window = words[max(0, index - 8) : index + 9]
        if not subjects.intersection(window):
            continue
        if not negations.intersection(words[max(0, index - 5) : index]):
            return True
    return False


def _has_word(text, *words):
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _evidence_line_obligations(text):
    """Recognize the visible semantic evidence obligations without fixed tokens."""
    folded = text.casefold()
    explains_binding = (
        _has_word(folded, "binder") and _has_word(folded, "parameter", "variable")
    ) or (
        _has_word(folded, "inner")
        and _has_word(folded, "hides", "shadows", "binds")
        and _has_word(folded, "parameter", "variable")
    )
    explains_predicate = (
        _has_word(folded, "whole")
        and _has_word(folded, "candidate", "set")
        and _has_word(folded, "sum", "sums")
        and _has_word(folded, "part", "subset", "subcollection", "proper")
    )
    explains_limitations = _has_word(folded, "lean") and any(
        phrase in folded
        for phrase in ("not assessed", "not verified", "not proved", "not proven")
    )
    return explains_binding, explains_predicate, explains_limitations


def _evidence_matches(evidence, result):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        marker_result = None
        explains_binding = False
        explains_predicate = False
        explains_limitations = False
        with target.open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("RESULT_JSON:"):
                    if marker_result is not None:
                        return False
                    marker_result = json.loads(
                        line.removeprefix("RESULT_JSON:").strip()
                    )
                    continue
                binding, predicate, limitations = _evidence_line_obligations(line)
                explains_binding |= binding
                explains_predicate |= predicate
                explains_limitations |= limitations
                if _has_affirmative_out_of_scope_claim(line):
                    return False
        return bool(
            marker_result == result
            and explains_binding
            and explains_predicate
            and explains_limitations
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ):
        return False


def main():
    submission = load_submission()
    source = _frozen_source()
    expected = json.loads((E / "expected.json").read_text())
    input_bound = workspace_input_is_bound(W / "input.json", tests=E)
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=ALLOWED_ASSURANCES,
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = bool(_valid_audit(result, source))
    evidence_valid = bool(
        isinstance(submission, dict)
        and _evidence_matches(submission.get("evidence"), submission.get("result"))
    )
    scope_correct = bool(
        isinstance(submission, dict)
        and type(submission.get("scope")) is str
        and submission.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = (
        submission.get("limitations") if isinstance(submission, dict) else None
    )
    limitations_correct = _limitations_valid(limitations)
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        contract
        and math_correct
        and limitations_correct
        and input_bound
        and not false_certification
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
                "input_binding": float(input_bound),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "limitation_accuracy": float(limitations_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
