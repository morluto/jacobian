from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import benchmarks.tooling.command_runner as command_runner_module
import benchmarks.tooling.lean_diagnostic_recovery as recovery_module
import pytest
from benchmarks.tooling.codex_visibility import surface_snapshot_digest
from benchmarks.tooling.lean_diagnostic_recovery import (
    RecoveryCase,
    classify_recovery,
    compare_report_paths,
    digest_suite,
    load_suite,
    summarize_runs,
)

from jacobian.canonical import canonicalize_json

ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / "benchmarks/config/lean-diagnostic-recovery-v1.json"
BASE_REVISION = "526575833ef9"
CANDIDATE_REVISION = "2" * 40


def _classify(
    case: RecoveryCase,
    telemetry: dict[str, object],
) -> dict[str, Any]:
    retained = dict(telemetry)
    invocations = telemetry.get("capability_invocations", [])
    retained.setdefault(
        "capability_attempts",
        [
            {
                "capability_id": invocation["capability_id"],
                "input": invocation["input"],
                "successful": True,
            }
            for invocation in invocations
        ],
    )
    return classify_recovery(case, retained)


def _surface(seed: str, deployed_revision: str) -> dict[str, object]:
    observed_revision = deployed_revision.ljust(40, "0")
    snapshot = {
        "server": {"name": "jacobian", "version": "0.11.0"},
        "instructions": f"test surface {seed}",
        "tools": [],
        "catalog": {
            "catalog_version": "1",
            "catalog_digest": "sha256:" + seed * 64,
            "policy_profile": "default",
            "policy_digest": "sha256:" + "9" * 64,
            "capability_count": 1,
            "content_sha256": "sha256:" + seed * 64,
        },
        "deployment": {
            "schema_version": "1",
            "revision": observed_revision,
            "package_version": "0.11.0",
            "evidence": "release-marker",
        },
    }
    return {**snapshot, "surface_digest": surface_snapshot_digest(snapshot)}


def _comparison_run(
    *,
    repair_success: bool,
    enriched_diagnostic_observed: bool,
    repeated_error_count: int,
    math_run_call_count: int,
    input_tokens: int,
    output_tokens: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "case_id": "core-check-type-mismatch",
        "repetition": 1,
        "command": {
            "status": "EXITED",
            "exit_code": 0,
            "elapsed_seconds": elapsed_seconds,
        },
        "metrics": {
            "injection_attempted": True,
            "injection_payload_exact": True,
            "injection_first_attempt": True,
            "injection_rejected": True,
            "observed_diagnostic_codes": [],
            "repair_success": repair_success,
            "enriched_diagnostic_observed": enriched_diagnostic_observed,
            "repeated_error_count": repeated_error_count,
            "repeated_mcp_call_count": 0,
            "math_run_call_count": math_run_call_count,
            "tool_error_count": 0,
            "tokens": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
        "artifacts": {
            "command": "core-check-type-mismatch-r01.command.json",
            "command_sha256": "sha256:" + "0" * 64,
            "transcript": "core-check-type-mismatch-r01.jsonl",
            "transcript_sha256": "sha256:" + "1" * 64,
            "stderr": "core-check-type-mismatch-r01.stderr",
            "stderr_sha256": "sha256:" + "2" * 64,
        },
    }


def _tool_event(
    capability_id: str,
    payload: dict[str, object],
    *,
    output: dict[str, object],
    verification_record_uri: str | None = None,
) -> dict[str, object]:
    response = {
        "capability_id": capability_id,
        "execution": {"status": "COMPLETED"},
        "output": output,
        "artifact_uris": [],
        "verification_record_uri": verification_record_uri,
    }
    return {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": "math.run",
            "arguments": {"capability_id": capability_id, "payload": payload},
            "status": "completed",
            "result": {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(response)}],
            },
        },
    }


def _comparison_evidence(
    condition: str,
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    case = load_suite(SUITE).cases[0]
    enriched = condition == "enriched-diagnostics"
    rejection = _tool_event(
        case.injected_capability_id,
        case.injected_payload,
        output={
            "conclusion": "UNKNOWN",
            "diagnostics": (
                [{"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}]
                if enriched
                else ["Lean rejected the proof: type mismatch"]
            ),
        },
    )
    if enriched:
        second = _tool_event(
            case.terminal_capability_id,
            {
                "statement": case.injected_payload["statement"],
                "proof": "by\n  trivial",
                "environment": case.injected_payload["environment"],
            },
            output={"conclusion": "TRUE", "diagnostics": []},
            verification_record_uri="artifact://sha256/" + "f" * 64,
        )
    else:
        second = rejection
    usage = {
        "input_tokens": 100 if enriched else 130,
        "output_tokens": 20 if enriched else 30,
    }
    transcript = (
        "\n".join(
            json.dumps(event)
            for event in (
                rejection,
                second,
                {"type": "turn.completed", "usage": usage},
            )
        )
        + "\n"
    ).encode()
    stderr = b""
    elapsed_seconds = 3.5 if enriched else 5.0
    command: dict[str, Any] = {
        "status": "EXITED",
        "exit_code": 0,
        "elapsed_seconds": elapsed_seconds,
    }
    command_receipt = canonicalize_json(
        {
            "status": command["status"],
            "exit_code": command["exit_code"],
            "elapsed_microseconds": round(elapsed_seconds * 1_000_000),
        }
    )
    return (
        {
            "case_id": case.case_id,
            "repetition": 1,
            "command": command,
            "metrics": {},
            "artifacts": {
                "command": "core-check-type-mismatch-r01.command.json",
                "command_sha256": "sha256:"
                + hashlib.sha256(command_receipt).hexdigest(),
                "transcript": "core-check-type-mismatch-r01.jsonl",
                "transcript_sha256": "sha256:" + hashlib.sha256(transcript).hexdigest(),
                "stderr": "core-check-type-mismatch-r01.stderr",
                "stderr_sha256": "sha256:" + hashlib.sha256(stderr).hexdigest(),
            },
        },
        transcript,
        stderr,
        command_receipt,
    )


def _comparison_report(
    condition: str,
    *,
    surface_seed: str | None = None,
) -> dict[str, object]:
    control = condition == "control"
    run, _transcript, _stderr, _command_receipt = _comparison_evidence(condition)
    # The fixture's metrics are the deterministic classification of the events
    # above; the comparator independently repeats this from the retained file.
    classified = {
        "injection_attempted": True,
        "injection_payload_exact": True,
        "injection_first_attempt": True,
        "injection_rejected": True,
        "observed_diagnostic_codes": ["LEAN_TYPE_MISMATCH"] if not control else [],
        "enriched_diagnostic_observed": not control,
        "repair_success": not control,
        "repeated_error_count": 1 if control else 0,
        "repeated_mcp_call_count": 1 if control else 0,
        "math_run_call_count": 2,
        "tool_error_count": 0,
        "tokens": {
            "input_tokens": 130 if control else 100,
            "output_tokens": 30 if control else 20,
        },
    }
    run["metrics"] = classified
    return {
        "schema_version": "1",
        "evidence_class": "public-host-local-lean-recovery-observation",
        "causal_claim_authorized": False,
        "suite_id": "lean-diagnostic-recovery-v1",
        "suite_digest": digest_suite(SUITE),
        "source_base_revision": BASE_REVISION,
        "source_candidate_revision": CANDIDATE_REVISION,
        "deployed_revision": BASE_REVISION if control else CANDIDATE_REVISION,
        "condition": condition,
        "model": "test-model",
        "reasoning_effort": "high",
        "tool_mode": "direct",
        "repetitions": 1,
        "timeout_seconds": 300.0,
        "codex_version": "codex-test",
        "selected_case_ids": ["core-check-type-mismatch"],
        "surface": _surface(
            surface_seed or ("b" if control else "c"),
            BASE_REVISION if control else CANDIDATE_REVISION,
        ),
        "runs": [run],
        "summary": summarize_runs([run]),
    }


def _write_comparison_reports(
    tmp_path: Path,
    control: dict[str, Any],
    treatment: dict[str, Any],
) -> tuple[Path, Path, str, str]:
    paths: list[Path] = []
    anchors: list[str] = []
    for slot, report in (("control", control), ("enriched-diagnostics", treatment)):
        root = tmp_path / slot
        root.mkdir(exist_ok=True)
        canonical_run, transcript, stderr, command_receipt = _comparison_evidence(slot)
        artifacts = canonical_run["artifacts"]
        (root / artifacts["command"]).write_bytes(command_receipt)
        (root / artifacts["transcript"]).write_bytes(transcript)
        (root / artifacts["stderr"]).write_bytes(stderr)
        report_path = root / "report.json"
        report_payload = json.dumps(report).encode()
        report_path.write_bytes(report_payload)
        paths.append(report_path)
        anchors.append("sha256:" + hashlib.sha256(report_payload).hexdigest())
    return paths[0], paths[1], anchors[0], anchors[1]


def _compare(
    tmp_path: Path,
    control: dict[str, Any],
    treatment: dict[str, Any],
) -> dict[str, object]:
    control_path, treatment_path, control_anchor, treatment_anchor = (
        _write_comparison_reports(tmp_path, control, treatment)
    )
    return compare_report_paths(
        control_path,
        treatment_path,
        control_report_sha256=control_anchor,
        treatment_report_sha256=treatment_anchor,
        suite_path=SUITE,
    )


def test_recovery_suite_freezes_control_treatment_and_injected_cases() -> None:
    suite = load_suite(SUITE)

    assert {condition.id for condition in suite.conditions} == {
        "control",
        "enriched-diagnostics",
    }
    assert len(suite.cases) == 5
    assert any("MATHLIB" in case.prompt for case in suite.cases)
    assert suite.cases[0].terminal_immutable_input_fields == (
        "statement",
        "environment",
    )
    assert suite.cases[2].terminal_immutable_input_fields == (
        "environment",
        "statement",
        "original_proof",
    )
    premise_probe = suite.cases[3].diagnostic_probe
    assert premise_probe is not None
    assert premise_probe.capability_id == "lean.retrieve.premises"
    assert premise_probe.expected_diagnostic_evidence.path == "proof_prefix.0"
    assert suite.cases[4].expected_diagnostic_evidence is not None
    assert suite.cases[4].expected_diagnostic_evidence.path == "$"
    assert suite.cases[4].expected_diagnostic_evidence.validation_error_paths == ("$",)
    assert suite.causal_claim_authorized is False


def test_recovery_classifies_failed_request_diagnostic_before_verified_repair() -> None:
    case = load_suite(SUITE).cases[3]
    probe = case.diagnostic_probe
    assert probe is not None
    terminal_input = {
        "environment": case.injected_payload["environment"],
        "statement": case.injected_payload["statement"],
        "proof": "by\n  intro x\n  exact sq_nonneg x",
    }
    result = classify_recovery(
        case,
        {
            "capability_attempts": [
                {
                    "capability_id": probe.capability_id,
                    "input": probe.payload,
                    "successful": False,
                    "diagnostic_codes": ["INVALID_LEAN_RETRIEVAL_REQUEST"],
                    "diagnostics": [
                        {
                            "code": "INVALID_LEAN_RETRIEVAL_REQUEST",
                            "path": "proof_prefix.0",
                            "details": {
                                "validation_errors": [
                                    {
                                        "path": "proof_prefix.0",
                                        "reason": "must not include by",
                                        "type": "value_error",
                                    }
                                ]
                            },
                        }
                    ],
                },
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "successful": True,
                },
                {
                    "capability_id": case.terminal_capability_id,
                    "input": terminal_input,
                    "successful": True,
                },
            ],
            "capability_invocations": [
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "output": {
                        "conclusion": "UNKNOWN",
                        "diagnostics": [
                            {
                                "code": "LEAN_UNKNOWN_IDENTIFIER",
                                "phase": "KERNEL_CHECK",
                            }
                        ],
                    },
                },
                {
                    "capability_id": case.terminal_capability_id,
                    "input": terminal_input,
                    "output": {"conclusion": "TRUE"},
                    "verification_record_uri": ("artifact://sha256/" + "a" * 64),
                },
            ],
            "mcp_calls": ["math.run", "math.run", "math.run"],
        },
    )

    assert result["injection_rejected"] is True
    assert result["observed_diagnostic_codes"] == ["LEAN_UNKNOWN_IDENTIFIER"]
    assert result["enriched_diagnostic_observed"] is True
    assert result["repair_success"] is True


def test_premise_probe_does_not_bias_control_repair_classification() -> None:
    case = load_suite(SUITE).cases[3]
    probe = case.diagnostic_probe
    assert probe is not None
    terminal_input = {
        "environment": case.injected_payload["environment"],
        "statement": case.injected_payload["statement"],
        "proof": "by\n  intro x\n  exact sq_nonneg x",
    }
    result = classify_recovery(
        case,
        {
            "capability_attempts": [
                {
                    "capability_id": probe.capability_id,
                    "input": probe.payload,
                    "successful": False,
                    "diagnostic_codes": ["LEAN_RETRIEVAL_FAILED"],
                },
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "successful": True,
                },
                {
                    "capability_id": case.terminal_capability_id,
                    "input": terminal_input,
                    "successful": True,
                },
            ],
            "capability_invocations": [
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "output": {
                        "conclusion": "UNKNOWN",
                        "diagnostics": ["Lean rejected the proof: unknown identifier"],
                    },
                },
                {
                    "capability_id": case.terminal_capability_id,
                    "input": terminal_input,
                    "output": {"conclusion": "TRUE"},
                    "verification_record_uri": "artifact://sha256/" + "b" * 64,
                },
            ],
        },
    )

    assert result["injection_rejected"] is True
    assert result["enriched_diagnostic_observed"] is False
    assert result["repair_success"] is True


@pytest.mark.parametrize(
    "generic_code",
    ("UNKNOWN_CAPABILITY", "INVALID_REQUEST", "ADAPTER_EXECUTION_FAILED"),
)
def test_recovery_does_not_credit_generic_failed_injection(
    generic_code: str,
) -> None:
    case = load_suite(SUITE).cases[0]
    terminal_input = {
        "environment": case.injected_payload["environment"],
        "statement": case.injected_payload["statement"],
        "proof": "by\n  intro x\n  exact sq_nonneg x",
    }
    result = classify_recovery(
        case,
        {
            "capability_attempts": [
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "successful": False,
                    "diagnostic_codes": [generic_code],
                },
                {
                    "capability_id": case.terminal_capability_id,
                    "input": terminal_input,
                    "successful": True,
                },
            ],
            "capability_invocations": [
                {
                    "capability_id": case.terminal_capability_id,
                    "input": terminal_input,
                    "output": {"conclusion": "TRUE"},
                    "verification_record_uri": "artifact://sha256/" + "a" * 64,
                }
            ],
        },
    )

    assert result["injection_rejected"] is False
    assert result["repair_success"] is False
    assert result["observed_diagnostic_codes"] == []


def test_recovery_uses_later_exact_retry_after_operational_failure() -> None:
    case = load_suite(SUITE).cases[0]
    repaired_input = {
        "statement": case.injected_payload["statement"],
        "proof": "by\n  trivial",
        "environment": case.injected_payload["environment"],
    }
    rejected = {
        "capability_id": case.injected_capability_id,
        "input": case.injected_payload,
        "output": {
            "conclusion": "UNKNOWN",
            "diagnostics": [{"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}],
        },
    }
    repaired = {
        "capability_id": case.terminal_capability_id,
        "input": repaired_input,
        "output": {"conclusion": "TRUE", "diagnostics": []},
        "verification_record_uri": "artifact://sha256/" + "b" * 64,
    }
    result = classify_recovery(
        case,
        {
            "capability_attempts": [
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "successful": False,
                    "diagnostic_codes": ["LEAN_CHECKER_TIMEOUT"],
                },
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "successful": True,
                },
                {
                    "capability_id": case.terminal_capability_id,
                    "input": repaired_input,
                    "successful": True,
                },
            ],
            "capability_invocations": [rejected, repaired],
        },
    )

    assert result["injection_first_attempt"] is True
    assert result["injection_rejected"] is True
    assert result["observed_diagnostic_codes"] == ["LEAN_TYPE_MISMATCH"]
    assert result["repair_success"] is True


def test_recovery_does_not_shift_invocations_after_malformed_success() -> None:
    case = load_suite(SUITE).cases[0]
    unrelated_input = {
        "statement": "False",
        "proof": "by\n  trivial",
        "environment": case.injected_payload["environment"],
    }
    repaired_input = {
        "statement": case.injected_payload["statement"],
        "proof": "by\n  trivial",
        "environment": case.injected_payload["environment"],
    }
    unrelated_rejection = {
        "capability_id": case.injected_capability_id,
        "input": unrelated_input,
        "output": {
            "conclusion": "UNKNOWN",
            "diagnostics": [{"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}],
        },
    }
    repaired = {
        "capability_id": case.terminal_capability_id,
        "input": repaired_input,
        "output": {"conclusion": "TRUE", "diagnostics": []},
        "verification_record_uri": "artifact://sha256/" + "d" * 64,
    }
    result = classify_recovery(
        case,
        {
            "capability_attempts": [
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "successful": True,
                },
                {
                    "capability_id": case.injected_capability_id,
                    "input": unrelated_input,
                    "successful": True,
                },
                {
                    "capability_id": case.terminal_capability_id,
                    "input": repaired_input,
                    "successful": True,
                },
            ],
            # The first response was malformed, so telemetry retained no
            # completed invocation for it. Later invocations must not shift.
            "capability_invocations": [unrelated_rejection, repaired],
        },
    )

    assert result["injection_payload_exact"] is True
    assert result["injection_rejected"] is False
    assert result["repair_success"] is False
    assert result["observed_diagnostic_codes"] == []


def test_enrichment_requires_expected_field_level_evidence() -> None:
    case = load_suite(SUITE).cases[4]
    terminal_input = {
        "environment": case.injected_payload["environment"],
        "statement": case.injected_payload["statement"],
        "proof": "by\n  trivial",
    }
    result = classify_recovery(
        case,
        {
            "capability_attempts": [
                {
                    "capability_id": case.injected_capability_id,
                    "input": case.injected_payload,
                    "successful": False,
                    "diagnostic_codes": ["INVALID_LEAN_TRANSITION_REQUEST"],
                    "diagnostics": [
                        {
                            "code": "INVALID_LEAN_TRANSITION_REQUEST",
                            "path": None,
                        }
                    ],
                },
                {
                    "capability_id": case.terminal_capability_id,
                    "input": terminal_input,
                    "successful": True,
                },
            ],
            "capability_invocations": [
                {
                    "capability_id": case.terminal_capability_id,
                    "input": terminal_input,
                    "output": {"conclusion": "TRUE"},
                    "verification_record_uri": "artifact://sha256/" + "c" * 64,
                }
            ],
        },
    )

    assert result["injection_rejected"] is True
    assert result["repair_success"] is True
    assert result["enriched_diagnostic_observed"] is False


def test_recovery_suite_bytes_have_a_stable_evaluation_identity() -> None:
    expected = "sha256:" + hashlib.sha256(SUITE.read_bytes()).hexdigest()

    assert digest_suite(SUITE) == expected


@pytest.mark.parametrize("porcelain", (b" M evaluator.py\n", b"M  telemetry.py\n"))
def test_recovery_source_preflight_rejects_unstaged_or_staged_changes(
    monkeypatch: pytest.MonkeyPatch,
    porcelain: bytes,
) -> None:
    def git_status(*args: Any, **kwargs: Any):
        return command_runner_module.ToolCommandResult(
            status=command_runner_module.ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=porcelain,
            stderr=b"",
        )

    monkeypatch.setattr(command_runner_module, "run_operator_command", git_status)

    assert command_runner_module.git_tracked_worktree_is_clean(ROOT) is False


def test_recovery_execution_refuses_a_dirty_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery_module,
        "git_tracked_worktree_is_clean",
        lambda _root: False,
    )

    with pytest.raises(SystemExit, match="clean tracked worktree"):
        recovery_module._candidate_revision(ROOT)


def test_recovery_binds_revision_observed_from_the_mcp_endpoint() -> None:
    surface = _surface("b", BASE_REVISION)

    observed = recovery_module._bind_observed_deployment_revision(
        surface,
        supplied_revision=BASE_REVISION,
        expected_revision=BASE_REVISION,
    )

    assert observed == BASE_REVISION.ljust(40, "0")


def test_recovery_rejects_a_stale_or_swapped_mcp_endpoint() -> None:
    stale_surface = _surface("b", "3" * 40)

    with pytest.raises(SystemExit, match="observed MCP deployment revision"):
        recovery_module._bind_observed_deployment_revision(
            stale_surface,
            supplied_revision=BASE_REVISION,
            expected_revision=BASE_REVISION,
        )


def test_recovery_classification_separates_diagnostic_from_terminal_success() -> None:
    case = load_suite(SUITE).cases[0]
    telemetry = {
        "mcp_calls": ["math.run", "math.run"],
        "repeated_mcp_call_count": 0,
        "tool_error_count": 0,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "capability_invocations": [
            {
                "capability_id": "lean.check",
                "input": case.injected_payload,
                "output": {
                    "conclusion": "UNKNOWN",
                    "diagnostics": [
                        {"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}
                    ],
                },
            },
            {
                "capability_id": "lean.check",
                "input": {
                    "statement": case.injected_payload["statement"],
                    "proof": "by\n  trivial",
                    "environment": case.injected_payload["environment"],
                },
                "output": {"conclusion": "TRUE", "diagnostics": []},
                "verification_record_uri": "artifact://sha256/" + "a" * 64,
            },
        ],
    }

    result = _classify(case, telemetry)

    assert result["injection_rejected"] is True
    assert result["injection_payload_exact"] is True
    assert result["injection_first_attempt"] is True
    assert result["enriched_diagnostic_observed"] is True
    assert result["repair_success"] is True
    assert result["math_run_call_count"] == 2
    assert result["repeated_error_count"] == 0


def test_recovery_does_not_count_verification_of_a_different_claim() -> None:
    case = load_suite(SUITE).cases[1]
    telemetry = {
        "capability_invocations": [
            {
                "capability_id": case.injected_capability_id,
                "input": case.injected_payload,
                "output": {
                    "conclusion": "UNKNOWN",
                    "diagnostics": [
                        {
                            "code": "LEAN_UNKNOWN_IDENTIFIER",
                            "phase": "KERNEL_CHECK",
                        }
                    ],
                },
            },
            {
                "capability_id": case.terminal_capability_id,
                "input": {
                    "statement": "True",
                    "proof": "by trivial",
                    "environment": "MATHLIB",
                },
                "output": {"conclusion": "TRUE", "diagnostics": []},
                "verification_record_uri": "artifact://sha256/" + "b" * 64,
            },
        ]
    }

    result = _classify(case, telemetry)

    assert result["injection_payload_exact"] is True
    assert result["repair_success"] is False


@pytest.mark.parametrize(
    ("diagnostic", "input_error"),
    (
        (
            {
                "code": "LEAN_TOOLCHAIN_SETUP_FAILED",
                "phase": "RUNTIME_SETUP",
            },
            "TOOLCHAIN_PROBE: pinned Lean is unavailable",
        ),
        (
            {
                "code": "LEAN_MATHLIB_SETUP_FAILED",
                "phase": "RUNTIME_SETUP",
            },
            "MATHLIB_MANIFEST: pinned Mathlib is unavailable",
        ),
        (
            "TOOLCHAIN_PROBE: pinned Lean is unavailable",
            "TOOLCHAIN_PROBE: pinned Lean is unavailable",
        ),
    ),
)
def test_recovery_excludes_operational_failures_from_repairs(
    diagnostic: object,
    input_error: str,
) -> None:
    case = load_suite(SUITE).cases[0]
    telemetry = {
        "capability_invocations": [
            {
                "capability_id": case.injected_capability_id,
                "input": case.injected_payload,
                "output": {
                    "conclusion": "UNKNOWN",
                    "diagnostics": [diagnostic],
                    "input": {"status": "REJECTED", "errors": [input_error]},
                },
            },
            {
                "capability_id": case.terminal_capability_id,
                "input": {
                    "statement": case.injected_payload["statement"],
                    "proof": "by\n  trivial",
                    "environment": case.injected_payload["environment"],
                },
                "output": {"conclusion": "TRUE", "diagnostics": []},
                "verification_record_uri": "artifact://sha256/" + "c" * 64,
            },
        ]
    }

    result = _classify(case, telemetry)

    assert result["injection_rejected"] is False
    assert result["repair_success"] is False


def test_repeated_error_identity_is_condition_independent() -> None:
    case = load_suite(SUITE).cases[0]

    def telemetry(*, enriched: bool) -> dict[str, object]:
        payloads = (
            case.injected_payload,
            {**case.injected_payload, "proof": "by\n  exact missing_name"},
            case.injected_payload,
        )
        diagnostics: tuple[list[object], ...] = (
            (
                [{"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}]
                if enriched
                else ["Lean rejected the proof: type mismatch"]
            ),
            (
                [{"code": "LEAN_UNKNOWN_IDENTIFIER", "phase": "KERNEL_CHECK"}]
                if enriched
                else ["Lean rejected the proof: unknown identifier"]
            ),
            (
                [{"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}]
                if enriched
                else ["Lean rejected the proof with different legacy formatting"]
            ),
        )
        return {
            "capability_invocations": [
                {
                    "capability_id": case.injected_capability_id,
                    "input": payload,
                    "output": {
                        "conclusion": "UNKNOWN",
                        "diagnostics": diagnostic,
                    },
                }
                for payload, diagnostic in zip(payloads, diagnostics, strict=True)
            ]
        }

    control = _classify(case, telemetry(enriched=False))
    treatment = _classify(case, telemetry(enriched=True))

    assert control["repeated_error_count"] == 1
    assert treatment["repeated_error_count"] == 1


def test_recovery_keeps_legacy_proof_edit_control_observable() -> None:
    case = load_suite(SUITE).cases[2]
    corrected = {**case.injected_payload, "edited_proof": "by\n  trivial"}
    telemetry = {
        "capability_invocations": [
            {
                "capability_id": case.injected_capability_id,
                "input": case.injected_payload,
                "output": {
                    "accepted": False,
                    "baseline_accepted": True,
                    "baseline_checker_execution_status": "COMPLETED",
                    "checker_execution_status": "COMPLETED",
                },
            },
            {
                "capability_id": case.terminal_capability_id,
                "input": corrected,
                "output": {"accepted": True},
                "verification_record_uri": "artifact://sha256/" + "d" * 64,
            },
        ]
    }

    result = _classify(case, telemetry)

    assert result["injection_rejected"] is True
    assert result["repair_success"] is True


def test_recovery_summary_and_comparison_keep_efficiency_metrics_separate(
    tmp_path: Path,
) -> None:
    runs = [
        _comparison_run(
            repair_success=True,
            enriched_diagnostic_observed=True,
            repeated_error_count=0,
            math_run_call_count=2,
            input_tokens=100,
            output_tokens=20,
            elapsed_seconds=3.5,
        )
    ]
    treatment_summary = summarize_runs(runs)
    assert treatment_summary["math_run_call_count"] == 2
    assert treatment_summary["injection_first_attempt_count"] == 1
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")

    compared = _compare(tmp_path, control, treatment)

    assert compared["deltas"]["repair_success_rate"] == 1.0
    assert compared["deltas"]["repeated_error_count"] == -1
    assert compared["causal_claim_authorized"] is False
    assert compared["report_sha256"]["control"].startswith("sha256:")
    assert compared["report_sha256"]["treatment"].startswith("sha256:")
    assert (
        compared["condition_bindings"]["control"]["deployed_revision"] == BASE_REVISION
    )
    assert (
        compared["condition_bindings"]["enriched-diagnostics"]["deployed_revision"]
        == CANDIDATE_REVISION
    )


def test_recovery_does_not_count_a_repaired_call_before_exact_injection() -> None:
    case = load_suite(SUITE).cases[0]
    telemetry = {
        "capability_invocations": [
            {
                "capability_id": "lean.check",
                "input": {"statement": "True", "proof": "by trivial"},
                "output": {"conclusion": "TRUE"},
                "verification_record_uri": "artifact://sha256/" + "a" * 64,
            },
            {
                "capability_id": "lean.check",
                "input": case.injected_payload,
                "output": {
                    "conclusion": "UNKNOWN",
                    "diagnostics": [
                        {"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}
                    ],
                },
            },
        ]
    }

    result = _classify(case, telemetry)

    assert result["injection_attempted"] is True
    assert result["injection_payload_exact"] is True
    assert result["injection_first_attempt"] is False
    assert result["injection_rejected"] is True
    assert result["repair_success"] is False


def test_recovery_protocol_includes_failed_math_run_attempts() -> None:
    case = load_suite(SUITE).cases[0]
    rejected = {
        "capability_id": case.injected_capability_id,
        "input": case.injected_payload,
        "output": {
            "conclusion": "UNKNOWN",
            "diagnostics": [{"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}],
        },
    }
    repaired = {
        "capability_id": case.terminal_capability_id,
        "input": {
            "statement": case.injected_payload["statement"],
            "proof": "by\n  trivial",
            "environment": case.injected_payload["environment"],
        },
        "output": {"conclusion": "TRUE", "diagnostics": []},
        "verification_record_uri": "artifact://sha256/" + "e" * 64,
    }
    telemetry = {
        "capability_attempts": [
            {
                "capability_id": None,
                "input": {"malformed": True},
                "successful": False,
            },
            {
                "capability_id": case.injected_capability_id,
                "input": case.injected_payload,
                "successful": True,
            },
            {
                "capability_id": case.terminal_capability_id,
                "input": repaired["input"],
                "successful": True,
            },
        ],
        "capability_invocations": [rejected, repaired],
    }

    result = classify_recovery(case, telemetry)

    assert result["injection_attempted"] is True
    assert result["injection_payload_exact"] is True
    assert result["injection_first_attempt"] is False
    assert result["injection_rejected"] is True
    assert result["repair_success"] is True


def test_recovery_success_allows_an_atomic_operation_before_injection() -> None:
    case = load_suite(SUITE).cases[0]
    rejected = {
        "capability_id": case.injected_capability_id,
        "input": case.injected_payload,
        "output": {
            "conclusion": "UNKNOWN",
            "diagnostics": [{"code": "LEAN_TYPE_MISMATCH", "phase": "KERNEL_CHECK"}],
        },
    }
    repaired_input = {
        "statement": case.injected_payload["statement"],
        "proof": "by\n  trivial",
        "environment": case.injected_payload["environment"],
    }
    repaired = {
        "capability_id": case.terminal_capability_id,
        "input": repaired_input,
        "output": {"conclusion": "TRUE", "diagnostics": []},
        "verification_record_uri": "artifact://sha256/" + "f" * 64,
    }
    unrelated = {
        "capability_id": "arithmetic.gcd",
        "input": {"values": [12, 18]},
        "output": {"gcd": "6"},
    }
    telemetry = {
        "capability_attempts": [
            {
                "capability_id": unrelated["capability_id"],
                "input": unrelated["input"],
                "successful": True,
            },
            {
                "capability_id": case.injected_capability_id,
                "input": case.injected_payload,
                "successful": True,
            },
            {
                "capability_id": case.terminal_capability_id,
                "input": repaired_input,
                "successful": True,
            },
        ],
        "capability_invocations": [unrelated, rejected, repaired],
    }

    result = classify_recovery(case, telemetry)

    assert result["injection_payload_exact"] is True
    assert result["injection_first_attempt"] is False
    assert result["injection_rejected"] is True
    assert result["repair_success"] is True


def test_recovery_comparison_rejects_model_drift(tmp_path: Path) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    with pytest.raises(ValueError, match="model"):
        _compare(tmp_path, control, {**treatment, "model": "second"})


@pytest.mark.parametrize(
    ("field", "changed", "message"),
    (
        ("timeout_seconds", 301.0, "timeout_seconds"),
        ("selected_case_ids", ["proof-edit-type-mismatch"], "selected_case_ids"),
        ("source_base_revision", "3" * 40, "source_base_revision"),
        ("source_candidate_revision", "4" * 40, "source_candidate_revision"),
    ),
)
def test_recovery_comparison_rejects_run_invariant_drift(
    tmp_path: Path,
    field: str,
    changed: object,
    message: str,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")

    with pytest.raises(ValueError, match=message):
        _compare(tmp_path, control, {**treatment, field: changed})


def test_recovery_comparison_rejects_mislabeled_conditions(tmp_path: Path) -> None:
    control = _comparison_report("control")

    with pytest.raises(ValueError, match="enriched-diagnostics condition"):
        _compare(
            tmp_path,
            control,
            {**control, "deployed_revision": CANDIDATE_REVISION},
        )


def test_recovery_comparison_binds_each_deployment_to_its_source_revision(
    tmp_path: Path,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")

    with pytest.raises(ValueError, match="source_base_revision"):
        _compare(tmp_path, {**control, "deployed_revision": "3" * 40}, treatment)


def test_recovery_comparison_rejects_the_same_observed_server_surface(
    tmp_path: Path,
) -> None:
    control = _comparison_report("control", surface_seed="b")
    treatment = _comparison_report("enriched-diagnostics", surface_seed="b")

    with pytest.raises(ValueError, match="same MCP surface"):
        _compare(tmp_path, control, treatment)


@pytest.mark.parametrize("runs", (None, []))
def test_recovery_comparison_requires_retained_runs(
    tmp_path: Path, runs: object
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")

    with pytest.raises(ValueError, match="retained runs"):
        _compare(tmp_path, {**control, "runs": runs}, treatment)


def test_recovery_comparison_rejects_a_stale_summary(tmp_path: Path) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    stale = {**control["summary"], "repair_success_rate": 1.0}

    with pytest.raises(ValueError, match="summary does not match retained runs"):
        _compare(tmp_path, {**control, "summary": stale}, treatment)


def test_recovery_comparison_requires_each_case_repetition_pair(
    tmp_path: Path,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    first_run = control["runs"][0]
    duplicate_runs = [first_run, first_run]
    selected = ["core-check-type-mismatch", "mathlib-check-unknown-identifier"]
    invalid_control = {
        **control,
        "selected_case_ids": selected,
        "runs": duplicate_runs,
        "summary": summarize_runs(duplicate_runs),
    }
    matching_treatment_plan = {**treatment, "selected_case_ids": selected}

    with pytest.raises(ValueError, match="exactly one run per case and repetition"):
        _compare(tmp_path, invalid_control, matching_treatment_plan)


def test_recovery_comparison_rejects_malformed_retained_metrics(
    tmp_path: Path,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    original = control["runs"][0]
    malformed_runs = []

    non_boolean_success = deepcopy(original)
    non_boolean_success["metrics"]["repair_success"] = 2
    malformed_runs.append(non_boolean_success)

    negative_call_count = deepcopy(original)
    negative_call_count["metrics"]["math_run_call_count"] = -1
    malformed_runs.append(negative_call_count)

    negative_tokens = deepcopy(original)
    negative_tokens["metrics"]["tokens"]["input_tokens"] = -1
    malformed_runs.append(negative_tokens)

    non_finite_elapsed = deepcopy(original)
    non_finite_elapsed["command"]["elapsed_seconds"] = float("nan")
    malformed_runs.append(non_finite_elapsed)

    for malformed in malformed_runs:
        with pytest.raises(ValueError, match="malformed retained runs"):
            _compare(tmp_path, {**control, "runs": [malformed]}, treatment)


def test_recovery_comparison_recomputes_the_surface_digest(tmp_path: Path) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    surface = deepcopy(control["surface"])
    surface["instructions"] = "tampered after observation"

    with pytest.raises(ValueError, match="surface digest does not match"):
        _compare(tmp_path, {**control, "surface": surface}, treatment)


def test_recovery_comparison_verifies_retained_artifact_digests(
    tmp_path: Path,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    control_path, treatment_path, control_anchor, treatment_anchor = (
        _write_comparison_reports(tmp_path, control, treatment)
    )
    transcript_name = control["runs"][0]["artifacts"]["transcript"]
    (control_path.parent / transcript_name).write_bytes(b"tampered transcript\n")

    with pytest.raises(ValueError, match="artifact digest does not match"):
        compare_report_paths(
            control_path,
            treatment_path,
            control_report_sha256=control_anchor,
            treatment_report_sha256=treatment_anchor,
            suite_path=SUITE,
        )


def test_recovery_comparison_classifies_the_hash_verified_transcript_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    control_path, treatment_path, control_anchor, treatment_anchor = (
        _write_comparison_reports(tmp_path, control, treatment)
    )
    original_verified_artifact = recovery_module._verified_artifact
    replaced = False

    def replace_transcript_after_verification(*args: Any, **kwargs: Any):
        nonlocal replaced
        path, payload = original_verified_artifact(*args, **kwargs)
        if not replaced and path.suffix == ".jsonl":
            path.write_text(
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 999, "output_tokens": 999},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            replaced = True
        return path, payload

    monkeypatch.setattr(
        recovery_module,
        "_verified_artifact",
        replace_transcript_after_verification,
    )

    compared = compare_report_paths(
        control_path,
        treatment_path,
        control_report_sha256=control_anchor,
        treatment_report_sha256=treatment_anchor,
        suite_path=SUITE,
    )

    assert replaced is True
    assert compared["deltas"]["input_tokens"] == -30


def test_recovery_comparison_binds_completion_to_the_command_receipt(
    tmp_path: Path,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    treatment["runs"][0]["command"] = {
        "status": "TIMED_OUT",
        "exit_code": None,
        "elapsed_seconds": 3.5,
    }
    control_path, treatment_path, control_anchor, treatment_anchor = (
        _write_comparison_reports(tmp_path, control, treatment)
    )

    with pytest.raises(ValueError, match="command metadata does not match"):
        compare_report_paths(
            control_path,
            treatment_path,
            control_report_sha256=control_anchor,
            treatment_report_sha256=treatment_anchor,
            suite_path=SUITE,
        )


def test_recovery_comparison_rejects_a_self_consistent_forged_command_receipt(
    tmp_path: Path,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    control_path, treatment_path, control_anchor, _ = _write_comparison_reports(
        tmp_path, control, treatment
    )
    run = treatment["runs"][0]
    command_path = treatment_path.parent / run["artifacts"]["command"]

    timed_out_receipt = canonicalize_json(
        {"status": "TIMED_OUT", "exit_code": None, "elapsed_microseconds": 3_500_000}
    )
    command_path.write_bytes(timed_out_receipt)
    run["command"] = {
        "status": "TIMED_OUT",
        "exit_code": None,
        "elapsed_seconds": 3.5,
    }
    run["artifacts"]["command_sha256"] = (
        "sha256:" + hashlib.sha256(timed_out_receipt).hexdigest()
    )
    run["metrics"]["repair_success"] = False
    treatment["summary"] = summarize_runs(treatment["runs"])
    trusted_payload = json.dumps(treatment).encode()
    treatment_path.write_bytes(trusted_payload)
    treatment_anchor = "sha256:" + hashlib.sha256(trusted_payload).hexdigest()

    forged_receipt = canonicalize_json(
        {"status": "EXITED", "exit_code": 0, "elapsed_microseconds": 3_500_000}
    )
    command_path.write_bytes(forged_receipt)
    run["command"] = {
        "status": "EXITED",
        "exit_code": 0,
        "elapsed_seconds": 3.5,
    }
    run["artifacts"]["command_sha256"] = (
        "sha256:" + hashlib.sha256(forged_receipt).hexdigest()
    )
    run["metrics"]["repair_success"] = True
    treatment["summary"] = summarize_runs(treatment["runs"])
    treatment_path.write_bytes(json.dumps(treatment).encode())

    with pytest.raises(ValueError, match="external SHA-256 anchor"):
        compare_report_paths(
            control_path,
            treatment_path,
            control_report_sha256=control_anchor,
            treatment_report_sha256=treatment_anchor,
            suite_path=SUITE,
        )


def test_recovery_comparison_reclassifies_hash_verified_transcripts(
    tmp_path: Path,
) -> None:
    control = _comparison_report("control")
    treatment = _comparison_report("enriched-diagnostics")
    control_path, treatment_path, _, treatment_anchor = _write_comparison_reports(
        tmp_path, control, treatment
    )
    transcript_name = control["runs"][0]["artifacts"]["transcript"]
    transcript_path = control_path.parent / transcript_name
    changed = (
        transcript_path.read_bytes()
        + json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 999, "output_tokens": 999},
            }
        ).encode()
        + b"\n"
    )
    transcript_path.write_bytes(changed)
    control["runs"][0]["artifacts"]["transcript_sha256"] = (
        "sha256:" + hashlib.sha256(changed).hexdigest()
    )
    control_payload = json.dumps(control).encode()
    control_path.write_bytes(control_payload)
    control_anchor = "sha256:" + hashlib.sha256(control_payload).hexdigest()

    with pytest.raises(ValueError, match="metrics do not match"):
        compare_report_paths(
            control_path,
            treatment_path,
            control_report_sha256=control_anchor,
            treatment_report_sha256=treatment_anchor,
            suite_path=SUITE,
        )
