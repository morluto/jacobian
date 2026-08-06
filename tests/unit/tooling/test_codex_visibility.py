from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.codex_visibility import (
    CueLevel,
    VisibilityCase,
    classify_visibility,
    load_suite,
)
from pydantic import ValidationError

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


def test_committed_visibility_suite_has_explicit_and_latent_cases() -> None:
    suite = load_suite(_ROOT / "benchmarks/config/codex-visibility-v1.json")

    assert {case.cue_level for case in suite.cases} == {
        CueLevel.EXPLICIT,
        CueLevel.AFFORDANCE,
        CueLevel.LATENT,
    }
    assert any(case.require_verified for case in suite.cases)


def test_packaged_codex_skill_matches_repository_skill() -> None:
    repository_skill = _ROOT / ".agents/skills/jacobian-math/SKILL.md"
    packaged_skill = _ROOT / "npm/skills/jacobian-math/SKILL.md"

    assert packaged_skill.read_bytes() == repository_skill.read_bytes()


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
    }
    assert result["expected_capabilities"]["missing_completed"] == []
    assert result["contract_satisfied"] is True
    assert result["shell_call_count"] == 1


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
