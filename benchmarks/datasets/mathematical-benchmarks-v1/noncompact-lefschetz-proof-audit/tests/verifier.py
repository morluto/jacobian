import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
LIMITATION = (
    "The original First Proof lattice/manifold question and any alternative "
    "proof are not assessed."
)


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


def _counterexample_is_valid(value: object) -> bool:
    fields = {
        "space",
        "translation",
        "fixed_point_equation",
        "fixed_point_exists",
        "proper",
        "compact_support_cohomology",
        "top_degree_action",
        "lefschetz_number",
    }
    if not isinstance(value, dict) or set(value) != fields:
        return False
    translation = value["translation"]
    if not isinstance(translation, dict) or set(translation) != {
        "numerator",
        "denominator",
    }:
        return False
    numerator = translation["numerator"]
    denominator = translation["denominator"]
    if (
        not _exact_int(numerator)
        or not _exact_int(denominator)
        or not -1_000_000 <= numerator <= 1_000_000
        or numerator == 0
        or not 1 <= denominator <= 1_000_000
    ):
        return False
    cohomology = value["compact_support_cohomology"]
    if not isinstance(cohomology, list) or len(cohomology) != 2:
        return False
    normalized_cohomology: dict[int, int] = {}
    for entry in cohomology:
        if not isinstance(entry, dict) or set(entry) != {"degree", "dimension"}:
            return False
        degree = entry["degree"]
        dimension = entry["dimension"]
        if (
            not _exact_int(degree)
            or not _exact_int(dimension)
            or degree in normalized_cohomology
        ):
            return False
        normalized_cohomology[degree] = dimension
    top_action = value["top_degree_action"]
    lefschetz = value["lefschetz_number"]
    if not _exact_int(top_action) or not _exact_int(lefschetz):
        return False
    fixed_point_exists = numerator == 0
    return bool(
        value["space"] == "R"
        and value["fixed_point_equation"] == "x+q=x"
        and value["fixed_point_exists"] is fixed_point_exists
        and value["proper"] is True
        and normalized_cohomology == {0: 0, 1: 1}
        and top_action == 1
        and lefschetz == -top_action
        and not fixed_point_exists
        and lefschetz != 0
    )


def _result_is_valid(result: object, source: dict) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "counterexample",
        "theorem_audit",
        "downstream_status",
        "original_question_status",
    }:
        return False
    audit = result["theorem_audit"]
    expected_audit = {
        "frozen_lemma_status": "FALSE",
        "invalid_step": "COMPACTLY_SUPPORTED_DIAGONAL_THOM_CLASS",
        "repair_boundary": "REQUIRES_ADDITIONAL_HYPOTHESES_OR_DIFFERENT_ARGUMENT",
    }
    scope = source.get("audit_scope", {})
    return bool(
        source.get("expert_counterexample", {}).get("space") == "R"
        and scope.get("compact_support_lefschetz_replay") is True
        and scope.get("original_research_question") == "NOT_ASSESSED"
        and _counterexample_is_valid(result["counterexample"])
        and audit == expected_audit
        and result["downstream_status"] == "UNSUPPORTED_BY_FROZEN_ARGUMENT"
        and result["original_question_status"] == "NOT_ASSESSED"
    )


def main() -> None:
    submission = load_submission()
    input_binding = workspace_input_is_bound()
    source = _load_frozen_input()
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = bool(submission is not None and _result_is_valid(result, source))
    reward = float(math_ok and input_binding)

    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
