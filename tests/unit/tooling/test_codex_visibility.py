from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
from benchmarks.tooling.codex_visibility import (
    AdoptionExpectation,
    CueLevel,
    ToolMode,
    VisibilityCase,
    _build_summary,
    _codex_arguments,
    _run_case,
    classify_visibility,
    load_suite,
)
from benchmarks.tooling.command_runner import ToolCommandResult, ToolCommandStatus
from pydantic import ValidationError

from jacobian.contracts.matrix_operations import MatrixDeterminantRequest
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


def test_unified_exec_mode_is_opt_in(tmp_path: Path) -> None:
    common = {
        "workspace": tmp_path,
        "model": "test-model",
        "reasoning_effort": "high",
        "mcp_url": "https://example.test/mcp",
        "prompt": "Compute exactly.",
    }

    direct = _codex_arguments(**common, tool_mode=ToolMode.DIRECT)
    unified = _codex_arguments(**common, tool_mode=ToolMode.UNIFIED_EXEC)

    assert "unified_exec" not in direct
    assert unified[-3:-1] == ("--enable", "unified_exec")


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


def test_codex_skill_routes_exact_outcomes_without_catalog_projection() -> None:
    skill = (_ROOT / ".agents/skills/jacobian-math/SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "tools.mcp__jacobian__math_find" in skill
    assert "tools.mcp__jacobian__math_run" in skill
    # Check for key phrases that may span multiple lines in the SKILL.md.
    # Normalize whitespace to avoid brittleness from line rewrapping.
    skill_flat = re.sub(r"\s+", " ", skill)
    assert "Do not enumerate, filter, or print `ALL_TOOLS`" in skill_flat
    assert "text(r.structuredContent ?? r)" in skill
    assert 'math.find({"capability_id":"<exact-id>","view":"CONTRACT"})' in skill
    assert 'never send `mode: "CONTRACT"` to `math.run`' in skill_flat
    assert "never reconstruct or paraphrase such a record" in skill_flat
    assert "required task authorization and bindings are preserved" in skill_flat
    for guidance in (
        "Keep decomposition and routing decisions agent-owned",
        "composing already-known supporting operations remains allowed",
        "follow those fields",
        "retry within the task resource bounds",
        "continue with other installed routes",
        "completeness, and open obligations",
    ):
        assert guidance in skill


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


def _patched_run_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    result: ToolCommandResult,
) -> dict[str, object]:
    """Run ``_run_case`` with ``run_operator_command`` replaced by a stub.

    The stub avoids real Codex execution while exercising the real timing,
    transcript write, telemetry parse, and classification paths.  An empty
    JSONL transcript is sufficient for ``parse_agent_transcript`` and yields a
    clean abstention classification for the USE case.
    """

    monkeypatch.setattr(
        "benchmarks.tooling.codex_visibility.run_operator_command",
        lambda *_args, **_kwargs: result,
    )
    return _run_case(
        case=_case(),
        repetition=1,
        workspace=tmp_path / "workspace",
        output=tmp_path,
        model="test-model",
        reasoning_effort="high",
        mcp_url="https://example.test/mcp",
        timeout_seconds=5.0,
        tool_mode=ToolMode.DIRECT,
    )


def test_run_case_records_elapsed_seconds_for_successful_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "workspace").mkdir()
    result = ToolCommandResult(
        status=ToolCommandStatus.EXITED,
        exit_code=0,
        stdout=b"",
        stderr=b"",
    )

    run = _patched_run_case(monkeypatch, tmp_path, result=result)

    elapsed = run["command"]["elapsed_seconds"]
    assert isinstance(elapsed, float)
    assert math.isfinite(elapsed)
    assert elapsed >= 0.0
    assert run["command"]["status"] == ToolCommandStatus.EXITED
    assert run["command"]["exit_code"] == 0


def test_run_case_records_elapsed_seconds_for_failed_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "workspace").mkdir()
    result = ToolCommandResult(
        status=ToolCommandStatus.START_FAILED,
        exit_code=None,
        stdout=b"",
        stderr=b"",
        diagnostic="codex is unavailable",
    )

    run = _patched_run_case(monkeypatch, tmp_path, result=result)

    elapsed = run["command"]["elapsed_seconds"]
    assert isinstance(elapsed, float)
    assert math.isfinite(elapsed)
    assert elapsed >= 0.0
    assert run["command"]["status"] == ToolCommandStatus.START_FAILED
    assert run["command"]["exit_code"] is None
    assert run["command"]["diagnostic"] == "codex is unavailable"


def test_run_case_elapsed_seconds_does_not_affect_classification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "workspace").mkdir()
    result = ToolCommandResult(
        status=ToolCommandStatus.EXITED,
        exit_code=0,
        stdout=b"",
        stderr=b"",
    )

    run = _patched_run_case(monkeypatch, tmp_path, result=result)

    assert "elapsed_seconds" not in run["classification"]
    assert "elapsed_seconds" not in run["classification"]["observed"]
    assert "elapsed_seconds" not in run["artifacts"]


def test_visibility_report_serializes_duration_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "workspace").mkdir()
    exited = ToolCommandResult(
        status=ToolCommandStatus.EXITED,
        exit_code=0,
        stdout=b"",
        stderr=b"",
    )
    failed = ToolCommandResult(
        status=ToolCommandStatus.START_FAILED,
        exit_code=None,
        stdout=b"",
        stderr=b"",
        diagnostic="codex is unavailable",
    )

    results = [exited, failed]
    monkeypatch.setattr(
        "benchmarks.tooling.codex_visibility.run_operator_command",
        lambda *_args, **_kwargs: results.pop(0),
    )
    runs = [
        _run_case(
            case=_case(),
            repetition=rep,
            workspace=tmp_path / "workspace",
            output=tmp_path,
            model="test-model",
            reasoning_effort="high",
            mcp_url="https://example.test/mcp",
            timeout_seconds=5.0,
            tool_mode=ToolMode.DIRECT,
        )
        for rep in (1, 2)
    ]

    summary = _build_summary(runs)
    report = {
        "schema_version": "2",
        "runs": runs,
        "summary": summary,
    }

    serialized = json.dumps(report, indent=2, sort_keys=True)
    parsed = json.loads(serialized)

    for run in parsed["runs"]:
        assert "elapsed_seconds" in run["command"]
        assert isinstance(run["command"]["elapsed_seconds"], (int, float))
        assert run["command"]["elapsed_seconds"] >= 0
        assert "elapsed_seconds" not in run["classification"]
    assert "duration_totals" in parsed["summary"]
    assert "elapsed_seconds" in parsed["summary"]["duration_totals"]
    assert isinstance(
        parsed["summary"]["duration_totals"]["elapsed_seconds"], (int, float)
    )
    assert parsed["summary"]["duration_totals"]["elapsed_seconds"] >= 0
    assert parsed["summary"]["run_count"] == 2
    assert parsed["summary"]["command_failure_count"] == 1
