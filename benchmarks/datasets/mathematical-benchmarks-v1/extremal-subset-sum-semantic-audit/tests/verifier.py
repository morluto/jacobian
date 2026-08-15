import itertools
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


def main():
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    source = _frozen_source()
    input_bound = workspace_input_is_bound(W / "input.json", tests=E)
    result = data.get("result")
    math_correct = bool(
        isinstance(submission, dict) and input_bound and _valid_audit(result, source)
    )
    correct = math_correct
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
