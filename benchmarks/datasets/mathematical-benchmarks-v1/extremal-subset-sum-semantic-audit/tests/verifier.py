import itertools
import json
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
        if "lean" in folded and "not assessed" in folded:
            has_lean_not_assessed = True
        if any(
            term in folded for term in ("verified", "proved", "proven", "confirmed")
        ) and ("lean" in folded or "asymptotic" in folded or "conjecture" in folded):
            return False
    return has_lean_not_assessed


def _evidence_matches(evidence, result):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        )
        prose = " ".join(
            line for line in text.splitlines() if not line.startswith("RESULT_JSON:")
        ).casefold()
        return bool(
            json.loads(marker) == result
            and "shadow" in prose
            and "subset" in prose
            and ("not assessed" in prose or "not verified" in prose)
        )
    except (
        OSError,
        StopIteration,
        UnicodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ):
        return False


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
    submission = load_submission(require_input_binding=False)
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
    result = raw.get("result") if isinstance(raw, dict) else None
    math_correct = bool(_valid_audit(result, source))
    evidence_valid = bool(
        isinstance(raw, dict)
        and _evidence_matches(raw.get("evidence"), raw.get("result"))
    )
    scope_correct = bool(
        contract
        and isinstance(raw, dict)
        and raw.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        isinstance(raw, dict)
        and raw.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = raw.get("limitations", []) if isinstance(raw, dict) else []
    limitations_correct = _limitations_valid(limitations)
    false_certification = false_verified_claim(raw, verification_record_bound=False)
    correct = bool(
        contract
        and math_correct
        and limitations_correct
        and input_bound
        and not false_certification
    )
    reward = (
        0.0
        if not correct or not evidence_valid
        else 0.8 + 0.1 * scope_correct + 0.1 * assurance_correct
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
