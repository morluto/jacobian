from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).parents[2]
CORPUS_DIR = REPO_ROOT / "benchmarks" / "research_challenges"
SCHEMA_PATH = CORPUS_DIR / "public_postdoc.schema.json"
SUITE_PATH = CORPUS_DIR / "public_postdoc_v1.json"
FRONTIER_SUITE_PATH = CORPUS_DIR / "public_postdoc_frontier_v1.json"
SUITE_PATHS = (SUITE_PATH, FRONTIER_SUITE_PATH)
PROMPT_PREFIX = (
    "Use Jacobian MCP, and do not use web search or external knowledge retrieval,"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_public_postdoc_suite_conforms_to_its_schema() -> None:
    schema = _read_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)

    for suite_path in SUITE_PATHS:
        errors = sorted(
            Draft202012Validator(
                schema,
                format_checker=FormatChecker(),
            ).iter_errors(_read_json(suite_path)),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )

        assert not errors, f"{suite_path.name}\n" + "\n".join(
            f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
            for error in errors
        )


def test_public_postdoc_suite_is_explicitly_answer_visible_and_unscored() -> None:
    for suite_path in SUITE_PATHS:
        suite = _read_json(suite_path)

        assert suite["purpose"] == "PUBLIC_ANSWER_VISIBLE_DIAGNOSTIC"
        assert suite["scored"] is False
        assert all(case["oracle"]["answer_visible"] for case in suite["cases"])
        assert all(
            case["contamination"] == "PUBLIC_ANSWER_VISIBLE" for case in suite["cases"]
        )


def test_public_postdoc_prompts_do_not_disclose_evaluator_sources() -> None:
    for suite_path in SUITE_PATHS:
        suite = _read_json(suite_path)

        for case in suite["cases"]:
            prompt = case["prompt"]
            assert prompt.startswith(PROMPT_PREFIX)
            assert "http://" not in prompt
            assert "https://" not in prompt
            assert all(source["url"] not in prompt for source in case["sources"])


def test_public_postdoc_frontier_prompts_do_not_require_hidden_context() -> None:
    suite = _read_json(FRONTIER_SUITE_PATH)

    for case in suite["cases"]:
        assert "given in the problem statement" not in case["prompt"].lower()


def test_public_postdoc_case_ids_and_tier_mix_are_stable() -> None:
    suite = _read_json(SUITE_PATH)
    cases = suite["cases"]
    ids = [case["challenge_id"] for case in cases]

    assert ids == [f"jcb-postdoc-{number:03d}" for number in range(1, 13)]
    assert len({case["title"] for case in cases}) == len(cases)
    assert Counter(case["tier"] for case in cases) == {
        "CLOSURE_CANDIDATE": 3,
        "COMPOSITIONAL_STRETCH": 4,
        "CAPABILITY_GAP_PROBE": 5,
    }


def test_public_postdoc_frontier_ids_and_tier_mix_are_stable() -> None:
    suite = _read_json(FRONTIER_SUITE_PATH)
    cases = suite["cases"]
    ids = [case["challenge_id"] for case in cases]

    assert ids == [f"jcb-postdoc-{number:03d}" for number in range(13, 19)]
    assert len({case["title"] for case in cases}) == len(cases)
    assert Counter(case["tier"] for case in cases) == {
        "CLOSURE_CANDIDATE": 1,
        "COMPOSITIONAL_STRETCH": 2,
        "CAPABILITY_GAP_PROBE": 3,
    }


def test_magma_implication_oracle_replays_the_minimal_order_two_model() -> None:
    suite = _read_json(SUITE_PATH)
    case = next(
        item for item in suite["cases"] if item["challenge_id"] == "jcb-postdoc-003"
    )
    table = ((0, 1), (0, 1))

    def operation(left: int, right: int) -> int:
        return table[left][right]

    for left in range(2):
        for right in range(2):
            assert operation(left, right) == operation(
                operation(operation(right, left), left),
                right,
            )
    assert operation(operation(0, 1), 1) != operation(operation(1, 0), 0)

    def singleton_operation(left: int, right: int) -> int:
        assert left == right == 0
        return 0

    assert singleton_operation(singleton_operation(0, 0), 0) == singleton_operation(
        singleton_operation(0, 0),
        0,
    )
    assert "two-element" in case["oracle"]["summary"]
    assert "order one" in " ".join(case["success_criteria"]).lower()
