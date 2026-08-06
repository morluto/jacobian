from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.codex_visibility import (
    AdoptionExpectation,
    CueLevel,
    VisibilityCase,
    classify_visibility,
    load_suite,
)
from pydantic import ValidationError

from jacobian.contracts.matrices import MatrixDeterminantRequest
from jacobian.contracts.number_theory import IntegerPairRequest
from jacobian.contracts.polynomial_operations import PolynomialGcdRequest
from jacobian.eval.telemetry import parse_agent_transcript

_ROOT = Path(__file__).resolve().parents[3]


def _write_transcript(path: Path, *events: object) -> dict[str, object]:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return parse_agent_transcript(path)


def _case(*, verified: bool = False) -> VisibilityCase:
    return VisibilityCase(
        case_id="exact-determinant",
        cue_level=CueLevel.LATENT,
        prompt="Compute an exact determinant.",
        expected_capability_ids=("matrix.determinant.compute",),
        require_verified=verified,
    )


def _mcp_event(
    tool: str,
    arguments: object,
    response: object,
    *,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": tool,
            "arguments": arguments,
            "status": status,
            "result": {
                "structured_content": response,
                "content": [{"type": "text", "text": json.dumps(response)}],
            },
        },
    }


def test_load_suite_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    suite = {
        "schema_version": "1",
        "suite_id": "visibility-v1",
        "cases": [
            {
                "case_id": "same-case",
                "cue_level": "LATENT",
                "prompt": "first",
                "expected_capability_ids": ["integer.compute.gcd"],
            },
            {
                "case_id": "same-case",
                "cue_level": "EXPLICIT",
                "prompt": "second",
                "expected_capability_ids": ["integer.compute.gcd"],
            },
        ],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")

    with pytest.raises(ValidationError, match="case_id values must be unique"):
        load_suite(path)


def test_committed_visibility_v1_suite_remains_loadable() -> None:
    suite = load_suite(_ROOT / "benchmarks/config/codex-visibility-v1.json")

    assert {case.cue_level for case in suite.cases} == {
        CueLevel.EXPLICIT,
        CueLevel.AFFORDANCE,
        CueLevel.LATENT,
    }
    assert any(case.require_verified for case in suite.cases)


def test_committed_visibility_v2_suite_covers_domains_and_abstention() -> None:
    suite = load_suite(_ROOT / "benchmarks/config/codex-visibility-v2.json")

    expected_ids = {
        capability_id
        for case in suite.cases
        for capability_id in case.expected_capability_ids
    }
    assert suite.schema_version == "2"
    assert {
        "integer.compute.gcd",
        "integer.compute.euler_totient",
        "matrix.determinant.compute",
        "matrix.rank.compute",
        "polynomial.compute.gcd",
    } <= expected_ids
    assert (
        sum(case.expectation is AdoptionExpectation.ABSTAIN for case in suite.cases)
        >= 2
    )


def test_packaged_codex_skill_matches_repository_skill() -> None:
    repository_skill = _ROOT / ".agents/skills/jacobian-math/SKILL.md"
    packaged_skill = _ROOT / "npm/skills/jacobian-math/SKILL.md"

    assert packaged_skill.read_bytes() == repository_skill.read_bytes()


def test_codex_skill_keeps_bounded_stable_direct_run_contracts() -> None:
    skill = (_ROOT / ".agents/skills/jacobian-math/SKILL.md").read_text(
        encoding="utf-8"
    )

    for capability_id in (
        "integer.compute.gcd",
        "matrix.determinant.compute",
        "matrix.rank.compute",
        "polynomial.compute.gcd",
        "matrix.determinant.verify",
    ):
        assert f"`{capability_id}`" in skill
    integer_payload = '{"left":"84","right":"30"}'
    matrix_payload = '{"matrix":{"domain":"QQ","entries":[[{"num":"1","den":"1"}]]}}'
    polynomial = (
        '{"polynomial_schema_version":"1","domain":"QQ","variables":["x"],'
        '"polynomial":{"terms":[{"coefficient":{"num":"1","den":"1"},'
        '"exponents":[2]}]}}'
    )
    assert integer_payload in skill
    assert matrix_payload in skill
    assert polynomial in skill
    IntegerPairRequest.model_validate_json(integer_payload)
    MatrixDeterminantRequest.model_validate_json(matrix_payload)
    polynomial_value = json.loads(polynomial)
    PolynomialGcdRequest.model_validate(
        {"left": polynomial_value, "right": polynomial_value}
    )
    assert len(skill.encode("utf-8")) <= 4 * 1024


def test_visibility_classification_records_adoption_without_grading_shell(
    tmp_path: Path,
) -> None:
    telemetry = _write_transcript(
        tmp_path / "trace.jsonl",
        _mcp_event(
            "math.find",
            {"query": "exact determinant"},
            {
                "kind": "discovery",
                "matches": [{"capability_id": "matrix.determinant.compute"}],
            },
        ),
        _mcp_event(
            "math.find",
            {
                "capability_id": "matrix.determinant.compute",
                "view": "CONTRACT",
            },
            {"kind": "capability"},
        ),
        _mcp_event(
            "math.run",
            {
                "capability_id": "matrix.determinant.compute",
                "payload": {},
            },
            {
                "capability_id": "matrix.determinant.compute",
                "execution": {"status": "COMPLETED"},
                "output": {"determinant": "7"},
                "assurance": {
                    "level": "COMPUTED",
                    "verification_record_uri": None,
                },
            },
        ),
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "python check.py"},
        },
    )

    result = classify_visibility(_case(), telemetry)

    assert result["observed"] == {
        "discovered": True,
        "inspected": True,
        "invoked": True,
        "completed": True,
        "verified": False,
        "discovery_free_invocation": False,
        "abstained": False,
    }
    assert result["expected_capabilities"]["missing_completed"] == []
    assert result["contract_satisfied"] is True
    assert result["shell_call_count"] == 1
    assert result["mcp_call_count"] == 3
    assert result["math_find_call_count"] == 2
    assert result["math_run_call_count"] == 1


def test_visibility_classification_records_discovery_free_invocation(
    tmp_path: Path,
) -> None:
    telemetry = _write_transcript(
        tmp_path / "trace.jsonl",
        _mcp_event(
            "math.run",
            {
                "capability_id": "matrix.determinant.compute",
                "payload": {},
            },
            {
                "capability_id": "matrix.determinant.compute",
                "execution": {"status": "COMPLETED"},
                "assurance": {"level": "COMPUTED"},
            },
        ),
    )

    result = classify_visibility(_case(), telemetry)

    assert result["observed"]["discovery_free_invocation"] is True
    assert result["contract_satisfied"] is True


def test_visibility_classification_requires_abstention_for_negative_case(
    tmp_path: Path,
) -> None:
    case = VisibilityCase(
        case_id="conceptual-definition",
        cue_level=CueLevel.LATENT,
        expectation=AdoptionExpectation.ABSTAIN,
        prompt="Define a square matrix.",
    )
    clean = _write_transcript(tmp_path / "clean.jsonl")
    searched = _write_transcript(
        tmp_path / "searched.jsonl",
        _mcp_event("math.find", {"query": "square matrix"}, {"matches": []}),
    )

    clean_result = classify_visibility(case, clean)
    searched_result = classify_visibility(case, searched)

    assert clean_result["observed"]["abstained"] is True
    assert clean_result["contract_satisfied"] is True
    assert searched_result["observed"]["abstained"] is False
    assert searched_result["contract_satisfied"] is False
    resource_result = classify_visibility(
        case,
        {"mcp_resource_read_attempts": 1},
    )
    assert resource_result["observed"]["abstained"] is False
    assert resource_result["contract_satisfied"] is False


def test_visibility_case_rejects_inconsistent_expectations() -> None:
    with pytest.raises(ValidationError, match="USE cases require"):
        VisibilityCase(
            case_id="missing-capability",
            cue_level=CueLevel.LATENT,
            prompt="Compute something.",
        )
    with pytest.raises(ValidationError, match="ABSTAIN cases cannot declare"):
        VisibilityCase(
            case_id="negative-with-capability",
            cue_level=CueLevel.LATENT,
            expectation=AdoptionExpectation.ABSTAIN,
            prompt="Define a matrix.",
            expected_capability_ids=("matrix.rank.compute",),
        )


def test_visibility_classification_requires_bound_verified_evidence(
    tmp_path: Path,
) -> None:
    telemetry = _write_transcript(
        tmp_path / "trace.jsonl",
        _mcp_event(
            "math.run",
            {
                "capability_id": "matrix.determinant.compute",
                "payload": {},
            },
            {
                "capability_id": "matrix.determinant.compute",
                "execution": {"status": "COMPLETED"},
                "assurance": {
                    "level": "VERIFIED",
                    "verification_record_uri": None,
                },
            },
        ),
    )

    result = classify_visibility(_case(verified=True), telemetry)

    assert result["observed"]["verified"] is False
    assert result["contract_satisfied"] is False


def test_visibility_classification_rejects_unrelated_verified_invocation() -> None:
    telemetry = {
        "capability_ids": ["matrix.determinant.compute"],
        "capability_attempt_ids": ["matrix.determinant.compute"],
        "capability_invocations": [
            {
                "capability_id": "integer.gcd.verify",
                "assurance": {
                    "level": "VERIFIED",
                    "verification_record_uri": "artifact://sha256/record",
                },
            }
        ],
    }

    result = classify_visibility(_case(verified=True), telemetry)

    assert result["observed"]["verified"] is False
    assert result["contract_satisfied"] is False


def test_visibility_classification_treats_timeout_as_non_completion(
    tmp_path: Path,
) -> None:
    telemetry = _write_transcript(
        tmp_path / "trace.jsonl",
        _mcp_event(
            "math.run",
            {
                "capability_id": "matrix.determinant.compute",
                "payload": {},
            },
            {
                "capability_id": "matrix.determinant.compute",
                "execution": {"status": "TIMEOUT"},
            },
        ),
    )

    result = classify_visibility(_case(), telemetry)

    assert result["observed"]["invoked"] is True
    assert result["observed"]["completed"] is False
    assert result["contract_satisfied"] is False
