from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from benchmarks.tooling.codex_visibility import (
    AdoptionExpectation,
    CueLevel,
    ToolMode,
    VisibilityCase,
    VisibilityOutputOutcome,
    _build_summary,
    _codex_arguments,
    _inspect_codex_skill_surface,
    _prepare_isolated_codex_environment,
    _run_case,
    classify_visibility,
    load_suite,
)
from benchmarks.tooling.command_runner import ToolCommandResult, ToolCommandStatus
from pydantic import ValidationError

from jacobian.contracts.domain_operations import InlineOperationOutput
from jacobian.contracts.lean import LeanDeclarationSearchOutput
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

    tracked_ids = {
        capability_id
        for case in suite.cases
        for capability_id in (
            case.expected_capability_ids + case.diagnostic_capability_ids
        )
    }
    assert suite.schema_version == "2"
    assert {
        "integer.compute.gcd",
        "integer.compute.euler_totient",
        "matrix.determinant.compute",
        "matrix.rank.compute",
        "polynomial.compute.gcd",
    } <= tracked_ids
    assert (
        sum(case.expectation is AdoptionExpectation.ABSTAIN for case in suite.cases)
        >= 2
    )


def test_committed_lean_usability_suite_covers_atomic_formal_tools() -> None:
    suite = load_suite(_ROOT / "benchmarks/config/lean-usability-v1.json")
    tracked_ids = {
        capability_id
        for case in suite.cases
        for capability_id in (
            case.expected_capability_ids + case.diagnostic_capability_ids
        )
    }

    assert suite.schema_version == "2"
    assert {
        "lean.check",
        "lean.declaration.inspect",
        "lean.declaration.search",
        "lean.proof_state.apply_tactic",
        "lean.retrieve.premises",
        "lean.term.apply",
    } <= tracked_ids
    assert any(case.expectation is AdoptionExpectation.ABSTAIN for case in suite.cases)


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

    assert "--approve-for-me" in direct
    assert "never" not in direct
    assert "unified_exec" not in direct
    assert unified[-3:-1] == ("--enable", "unified_exec")
    assert 'mcp_servers.jacobian.default_tools_approval_mode="approve"' in unified
    assert "--ignore-user-config" in direct
    assert "--ignore-rules" in direct


def test_codex_isolation_copies_only_authentication(tmp_path: Path) -> None:
    source_home = tmp_path / "source-home"
    source_codex_home = source_home / ".codex"
    source_codex_home.mkdir(parents=True)
    (source_codex_home / "auth.json").write_text("{}", encoding="utf-8")
    (source_codex_home / "config.toml").write_text(
        "model = 'ambient'", encoding="utf-8"
    )
    (source_codex_home / "skills").mkdir()

    environment, isolation = _prepare_isolated_codex_environment(
        tmp_path / "isolation",
        source_environment={
            "HOME": str(source_home),
            "CODEX_HOME": str(source_codex_home),
            "PATH": "/bin",
        },
    )

    isolated_codex_home = Path(environment["CODEX_HOME"])
    assert environment["HOME"] != str(source_home)
    assert isolated_codex_home != source_codex_home
    assert sorted(path.name for path in isolated_codex_home.iterdir()) == ["auth.json"]
    assert isolation == {
        "schema_version": "1",
        "home_isolated": True,
        "codex_home_isolated": True,
        "user_config_loaded": False,
        "user_rules_loaded": False,
        "authentication_seeded": True,
    }


def test_codex_skill_surface_records_model_visible_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolated_home = tmp_path / "home"
    codex_home = tmp_path / "codex-home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt = f"""<skills_instructions>
## Skills
- imagegen: Generate images. (file: {codex_home}/skills/.system/imagegen/SKILL.md)
- leantoken: Explore repositories. (file: /home/operator/.agents/skills/leantoken/SKILL.md)
</skills_instructions>"""
    result = ToolCommandResult(
        status=ToolCommandStatus.EXITED,
        exit_code=0,
        stdout=json.dumps(
            [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ]
        ).encode(),
        stderr=b"",
    )
    monkeypatch.setattr(
        "benchmarks.tooling.codex_visibility.run_operator_command",
        lambda *_args, **_kwargs: result,
    )

    surface = _inspect_codex_skill_surface(
        workspace,
        {"HOME": str(isolated_home), "CODEX_HOME": str(codex_home)},
    )

    assert [skill["name"] for skill in surface["skills"]] == [
        "imagegen",
        "leantoken",
    ]
    assert surface["skills"][0]["source"].startswith("$CODEX_HOME/")
    assert surface["external_file_sources"] == [
        "/home/operator/.agents/skills/leantoken/SKILL.md"
    ]


def test_codex_skill_surface_rejects_unparsed_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = ToolCommandResult(
        status=ToolCommandStatus.EXITED,
        exit_code=0,
        stdout=json.dumps(
            [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "<skills_instructions>\n## Skills\n"
                                "- unknown format without a source\n"
                                "</skills_instructions>"
                            ),
                        }
                    ],
                }
            ]
        ).encode(),
        stderr=b"",
    )
    monkeypatch.setattr(
        "benchmarks.tooling.codex_visibility.run_operator_command",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(RuntimeError, match="unknown format"):
        _inspect_codex_skill_surface(
            tmp_path,
            {"HOME": str(tmp_path / "home"), "CODEX_HOME": str(tmp_path / "codex")},
        )


def test_visibility_classification_records_adoption_without_grading_shell(
    tmp_path: Path,
) -> None:
    telemetry = _write_transcript(
        tmp_path / "trace.jsonl",
        _mcp_event(
            "math.find",
            {
                "request": {
                    "op": "search",
                    "query": "exact determinant",
                }
            },
            {
                "kind": "discovery",
                "matches": [{"capability_id": "matrix.determinant.compute"}],
            },
        ),
        _mcp_event(
            "math.find",
            {
                "request": {
                    "op": "inspect",
                    "capability_id": "matrix.determinant.compute",
                }
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
                "verification_record_uri": None,
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
    assert result["empty_payload_probe_count"] == 0


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
                "verification_record_uri": None,
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
    with pytest.raises(ValidationError, match="cannot declare operations or outcomes"):
        VisibilityCase(
            case_id="negative-with-diagnostic-capability",
            cue_level=CueLevel.LATENT,
            expectation=AdoptionExpectation.ABSTAIN,
            prompt="Define a matrix.",
            diagnostic_capability_ids=("matrix.rank.compute",),
        )
    with pytest.raises(ValidationError, match="must be tracked"):
        VisibilityCase(
            case_id="untracked-outcome",
            cue_level=CueLevel.LATENT,
            prompt="Find a declaration.",
            acceptable_output_outcomes=(
                VisibilityOutputOutcome(
                    capability_id="lean.declaration.search",
                    required_output_fields=("declarations.0.name",),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="must be disjoint"):
        VisibilityCase(
            case_id="overlapping-capabilities",
            cue_level=CueLevel.LATENT,
            prompt="Compute a rank.",
            expected_capability_ids=("matrix.rank.compute",),
            diagnostic_capability_ids=("matrix.rank.compute",),
        )


def test_diagnostic_operation_observation_does_not_gate_outcome_contract() -> None:
    case = VisibilityCase(
        case_id="verified-proof-with-optional-term-transition",
        cue_level=CueLevel.AFFORDANCE,
        prompt="Prove and verify a theorem.",
        expected_capability_ids=("lean.check",),
        diagnostic_capability_ids=("lean.term.apply",),
        require_verified=True,
    )
    telemetry = {
        "capability_attempt_ids": ["lean.check"],
        "capability_ids": ["lean.check"],
        "capability_invocations": [
            {
                "capability_id": "lean.check",
                "verification_record_uri": "artifact://sha256/record",
            }
        ],
    }

    result = classify_visibility(case, telemetry)

    assert result["contract_satisfied"] is True
    assert result["expected_capabilities"]["missing_completed"] == []
    assert result["diagnostic_capabilities"]["not_completed"] == ["lean.term.apply"]
    assert result["unexpected_capabilities"]["completed"] == []


def test_declaration_outcome_accepts_native_inline_search_without_inspect() -> None:
    case = load_suite(_ROOT / "benchmarks/config/lean-usability-v1.json").cases[3]
    native_output = (
        InlineOperationOutput[LeanDeclarationSearchOutput]
        .model_validate(
            {
                "result": {
                    "environment": "MATHLIB",
                    "environment_digest": "sha256:" + "a" * 64,
                    "lean_version": "4.31.0",
                    "lean_commit": "lean-commit",
                    "mathlib_commit": "mathlib-commit",
                    "query": {
                        "environment": "MATHLIB",
                        "name_contains": "irrational_sqrt_two",
                        "kinds": ["THEOREM"],
                        "result_limit": 1,
                    },
                    "declarations": [
                        {
                            "name": "irrational_sqrt_two",
                            "type": "Irrational √2",
                            "kind": "THEOREM",
                            "match_reasons": ["NAME_SUBSTRING"],
                        }
                    ],
                    "scanned_declarations": 42,
                    "stop_reason": "RESULT_LIMIT",
                },
                "backend_version": "Lean 4.31.0 + Mathlib",
            }
        )
        .model_dump(mode="json")
    )
    telemetry = {
        "capability_attempt_ids": ["lean.declaration.search"],
        "capability_ids": ["lean.declaration.search"],
        "capability_invocations": [
            {
                "capability_id": "lean.declaration.search",
                "output": native_output,
            }
        ],
    }

    result = classify_visibility(case, telemetry)

    assert result["contract_satisfied"] is True
    assert result["expected_capabilities"]["missing_completed"] == []
    assert result["diagnostic_capabilities"]["not_completed"] == [
        "lean.declaration.inspect"
    ]
    assert result["output_outcomes"] == {
        "required": True,
        "satisfied": True,
        "matched_capability_ids": ["lean.declaration.search"],
    }


def test_declaration_outcome_rejects_incomplete_structured_output() -> None:
    case = load_suite(_ROOT / "benchmarks/config/lean-usability-v1.json").cases[3]
    result = classify_visibility(
        case,
        {
            "capability_attempt_ids": ["lean.declaration.search"],
            "capability_ids": ["lean.declaration.search"],
            "capability_invocations": [
                {
                    "capability_id": "lean.declaration.search",
                    "output": {
                        "result": {
                            "environment_digest": "sha256:" + "a" * 64,
                            "lean_version": "4.31.0",
                            "lean_commit": "lean-commit",
                            "mathlib_commit": "mathlib-commit",
                            "declarations": [{"name": "irrational_sqrt_two"}],
                        },
                        "backend_version": "Lean 4.31.0 + Mathlib",
                    },
                }
            ],
        },
    )

    assert result["contract_satisfied"] is False
    assert result["output_outcomes"]["satisfied"] is False


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
                "verification_record_uri": None,
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
                "verification_record_uri": "artifact://sha256/record",
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
        environment={},
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
            environment={},
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
    assert parsed["summary"]["recovery_totals"] == {
        "empty_payload_probe_count": 0,
        "failed_operation_attempt_count": 0,
        "repeated_error_count": 0,
    }
    assert parsed["summary"]["case_repetition_metrics"] == [
        {
            "case_id": "exact-determinant",
            "run_count": 2,
            "command_failure_count": 1,
            "contract_satisfied_count": 0,
            "contract_satisfaction_rate": 0.0,
            "empty_payload_probe_count": 0,
            "runs_with_empty_payload_probe": 0,
            "empty_payload_probe_run_rate": 0.0,
            "failed_operation_attempt_count": 0,
            "repeated_error_count": 0,
            "repeated_error_rate": None,
        }
    ]
