from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from benchmarks.tooling.command_runner import ToolCommandStatus
from benchmarks.tooling.trajectory_value_study import (
    TrajectoryValueStudySpec,
    _codex_arguments,
    _corpus,
    _required_reasoning_log,
    _submission_schema,
    _terminal_evidence,
    load_spec,
    run_study,
)
from benchmarks.tooling.trajectory_value_study_verifier import (
    MAX_RATIONAL_COMPONENT_DIGITS,
    file_digest,
    verify_workspace,
)
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian.eval.trajectory_state import (
    CleanRoomTerminalEvidence,
    TerminalAcceptance,
    extract_codex_trajectory,
)

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "benchmarks/config/trajectory-value-study-v1.json"
STUDY_PATH = ROOT / "benchmarks/studies/trajectory-state-value-codex-v1"


def _task(spec: TrajectoryValueStudySpec, task_id: str):
    return next(task for task in spec.tasks if task.task_id == task_id)


def _task_payload(task) -> dict[str, object]:
    return {
        "schema_version": "1",
        "task_id": task.task_id,
        "task_group": task.task_group,
        "task_family": task.task_family,
        "kind": task.kind,
        "statement": task.statement,
        "payload": task.payload,
    }


def _verify(tmp_path: Path, task, answer: object):
    payload = _task_payload(task)
    (tmp_path / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "submission.json").write_text(
        json.dumps({"task_id": task.task_id, "answer": answer}),
        encoding="utf-8",
    )
    return verify_workspace(payload, tmp_path)


def _reasoning_event(summary: str) -> dict[str, object]:
    return {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "tool": "reasoning.write",
            "arguments": {"phase": "PLAN", "summary": summary},
            "status": "completed",
            "result": {
                "structured_content": {"run_id": "run"},
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"run_id": "run"}),
                    }
                ],
            },
        },
    }


def _write_labelled_extraction(
    output: Path,
    *,
    trajectory_id: str,
    task_family: str,
    accepted: bool,
) -> None:
    run_dir = output / "runs" / trajectory_id
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript = run_dir / "codex.jsonl"
    transcript.write_text(
        json.dumps(_reasoning_event(f"Plan for {trajectory_id}")) + "\n",
        encoding="utf-8",
    )
    source_digest = file_digest(transcript)
    evidence = CleanRoomTerminalEvidence(
        verifier_digest="sha256:" + "1" * 64,
        source_binding_digest=source_digest,
        clean_room=True,
        verifier_execution_status="COMPLETED",
        acceptance=(
            TerminalAcceptance.ACCEPTED if accepted else TerminalAcceptance.REJECTED
        ),
        input_binding_valid=True,
        artifact_binding_valid=True,
    )
    extraction = extract_codex_trajectory(
        transcript, task_family=task_family, terminal_evidence=evidence
    )
    (run_dir / "extraction.json").write_text(
        json.dumps(extraction.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_frozen_spec_covers_four_families_and_repeated_rollouts() -> None:
    spec = load_spec(SPEC_PATH)

    assert spec.model.model_id == "gpt-5.4-mini"
    assert spec.model.reasoning_effort == "medium"
    assert spec.repetitions_per_task == 4
    assert len({task.task_family for task in spec.tasks}) == 4
    assert spec.reasoning_log_mode == "REQUIRED"
    assert spec.training_performed is False
    assert spec.scorer_intervention is False
    assert spec.exact_resume_supported is False


def test_committed_study_schema_matches_authoritative_contract() -> None:
    path = (
        ROOT
        / "docs/reference/evaluations/schemas/trajectory-value-study-v1.schema.json"
    )

    assert json.loads(path.read_text(encoding="utf-8")) == (
        TrajectoryValueStudySpec.model_json_schema(mode="validation")
    )


@pytest.mark.parametrize(
    ("task_id", "answer"),
    [
        (
            "integer-bezout-01",
            {"gcd": "99", "left_coefficient": "21", "right_coefficient": "-100"},
        ),
        ("matrix-determinant-01", {"determinant": "4312753"}),
        (
            "polynomial-gcd-bezout-01",
            {
                "gcd": ["1", "1", "1"],
                "left_bezout": ["-1/23", "4/23", "3/46"],
                "right_bezout": ["-13/46", "7/46", "-5/46", "-3/46"],
            },
        ),
        (
            "graph-independent-set-01",
            {"vertices": ["c", "e", "h", "j", "l"], "optimum": 5},
        ),
    ],
)
def test_clean_room_verifier_accepts_exact_alternate_witnesses(
    tmp_path: Path, task_id: str, answer: object
) -> None:
    spec = load_spec(SPEC_PATH)

    result = _verify(tmp_path, _task(spec, task_id), answer)

    assert result["acceptance"] == "ACCEPTED"
    assert result["input_binding_valid"] is True
    assert result["artifact_binding_valid"] is True
    assert result["clean_room"] is True
    assert all(result["checks"].values())


def test_verifier_rejects_wrong_bound_answer_without_losing_bindings(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "matrix-determinant-01")

    result = _verify(tmp_path, task, {"determinant": "0"})

    assert result["acceptance"] == "REJECTED"
    assert result["input_binding_valid"] is True
    assert result["artifact_binding_valid"] is True
    assert result["checks"]["exact_relation"] is False


def test_verifier_treats_missing_artifact_as_inconclusive(tmp_path: Path) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "integer-bezout-01")
    payload = _task_payload(task)
    (tmp_path / "task.json").write_text(json.dumps(payload), encoding="utf-8")

    result = verify_workspace(payload, tmp_path)

    assert result["acceptance"] == "INCONCLUSIVE"
    assert result["artifact_binding_valid"] is False


def test_verifier_rejects_substituted_task_binding(tmp_path: Path) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "integer-bezout-01")
    payload = _task_payload(task)
    substituted = json.loads(json.dumps(payload))
    substituted["payload"]["left"] = "1"
    (tmp_path / "task.json").write_text(json.dumps(substituted), encoding="utf-8")
    (tmp_path / "submission.json").write_text("{}", encoding="utf-8")

    result = verify_workspace(payload, tmp_path)

    assert result["acceptance"] == "INCONCLUSIVE"
    assert result["input_binding_valid"] is False
    assert result["artifact_binding_valid"] is True


def test_task_owned_submission_schemas_validate_known_witnesses() -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "graph-independent-set-01")
    schema = _submission_schema(task)

    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(
        {
            "task_id": task.task_id,
            "answer": {"vertices": ["c", "e", "h", "j", "l"], "optimum": 5},
        }
    )
    assert not Draft202012Validator(schema).is_valid(
        {
            "task_id": task.task_id,
            "answer": {"vertices": ["c", "c"], "optimum": 2},
        }
    )


def test_study_contract_is_closed_and_requires_frozen_kind_order() -> None:
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrajectoryValueStudySpec.model_validate(payload)

    payload.pop("unexpected")
    payload["tasks"] = list(reversed(payload["tasks"]))
    with pytest.raises(ValidationError, match="four frozen kinds in order"):
        TrajectoryValueStudySpec.model_validate(payload)


def test_codex_command_binds_exact_model_isolation_and_required_server(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC_PATH)
    arguments = _codex_arguments(
        workspace=tmp_path,
        spec=spec,
        mcp_url="http://127.0.0.1:8765/mcp",
        prompt="frozen prompt",
    )

    assert arguments[arguments.index("-m") + 1] == "gpt-5.4-mini"
    assert 'model_reasoning_effort="medium"' in arguments
    assert 'web_search="disabled"' in arguments
    assert 'mcp_servers.jacobian.url="http://127.0.0.1:8765/mcp"' in arguments
    assert "--ephemeral" in arguments
    assert "--ignore-user-config" in arguments
    assert "--ignore-rules" in arguments
    assert arguments[arguments.index("-s") + 1] == "workspace-write"
    assert arguments[-1] == "frozen prompt"


def test_clean_room_verifier_has_only_standard_library_imports() -> None:
    path = ROOT / "benchmarks/tooling/trajectory_value_study_verifier.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert roots <= {
        "__future__",
        "fractions",
        "hashlib",
        "itertools",
        "json",
        "math",
        "pathlib",
        "re",
        "stat",
        "typing",
    }


def test_verifier_rejects_duplicate_json_keys_before_binding(tmp_path: Path) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "integer-bezout-01")
    payload = _task_payload(task)
    (tmp_path / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "submission.json").write_text(
        '{"task_id":"integer-bezout-01","answer":{},"answer":{"gcd":"99"}}',
        encoding="utf-8",
    )

    result = verify_workspace(payload, tmp_path)

    assert result["acceptance"] == "REJECTED"
    assert result["input_binding_valid"] is True
    assert result["artifact_binding_valid"] is True
    assert result["checks"]["submission_json"] is False


def test_verifier_treats_duplicate_task_keys_as_unbound_input(tmp_path: Path) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "integer-bezout-01")
    payload = _task_payload(task)
    (tmp_path / "task.json").write_text(
        '{"schema_version":"1","schema_version":"1"}',
        encoding="utf-8",
    )
    (tmp_path / "submission.json").write_text("{}", encoding="utf-8")

    result = verify_workspace(payload, tmp_path)

    assert result["acceptance"] == "INCONCLUSIVE"
    assert result["input_binding_valid"] is False


def test_polynomial_verifier_rejects_oversized_rational_coefficients(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "polynomial-gcd-bezout-01")
    huge_denominator = "1" + "0" * MAX_RATIONAL_COMPONENT_DIGITS + "1"
    result = _verify(
        tmp_path,
        task,
        {
            "gcd": ["1"],
            "left_bezout": [f"1/{huge_denominator}"],
            "right_bezout": ["0"],
        },
    )

    assert result["acceptance"] == "REJECTED"
    assert result["input_binding_valid"] is True
    assert result["checks"]["answer_shape"] is False


def test_required_reasoning_log_failures_are_not_silently_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="ambiguous run identity"):
        _required_reasoning_log("http://127.0.0.1:1/mcp", ())
    with pytest.raises(RuntimeError, match="ambiguous run identity"):
        _required_reasoning_log("http://127.0.0.1:1/mcp", ("a", "b"))

    async def fail_read(_url: str, _run_id: str) -> str:
        raise RuntimeError("resource missing")

    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_study._read_reasoning_resource",
        fail_read,
    )
    with pytest.raises(RuntimeError, match="resource missing"):
        _required_reasoning_log("http://127.0.0.1:1/mcp", ("run",))


def test_terminal_evidence_binds_transcript_digest_and_strict_booleans() -> None:
    verifier = {
        "verifier_digest": "sha256:" + "1" * 64,
        "verifier_execution_status": "COMPLETED",
        "acceptance": "REJECTED",
        "input_binding_valid": 1,
        "artifact_binding_valid": "true",
    }

    terminal = _terminal_evidence(
        ToolCommandStatus.EXITED,
        0,
        verifier,
        "sha256:" + "2" * 64,
    )

    assert terminal.source_binding_digest == "sha256:" + "2" * 64
    assert terminal.input_binding_valid is False
    assert terminal.artifact_binding_valid is False


def test_model_execution_is_explicitly_opt_in(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="without --execute"):
        run_study(SPEC_PATH, tmp_path / "results", execute=False)


def test_singleton_labelled_task_groups_are_recorded_as_exclusions(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC_PATH)
    singleton_task = _task(spec, "integer-bezout-01")
    repeated_task = _task(spec, "matrix-determinant-01")
    output = tmp_path / "study"
    records = [
        {
            "trajectory_id": "integer-bezout-01-r01",
            "task_id": singleton_task.task_id,
            "terminal": {"acceptance": "ACCEPTED"},
        },
        {
            "trajectory_id": "integer-bezout-01-r02",
            "task_id": singleton_task.task_id,
            "terminal": {"acceptance": "INCONCLUSIVE"},
        },
        {
            "trajectory_id": "matrix-determinant-01-r01",
            "task_id": repeated_task.task_id,
            "terminal": {"acceptance": "ACCEPTED"},
        },
        {
            "trajectory_id": "matrix-determinant-01-r02",
            "task_id": repeated_task.task_id,
            "terminal": {"acceptance": "REJECTED"},
        },
    ]
    for record in records:
        if record["terminal"]["acceptance"] == "INCONCLUSIVE":
            continue
        task = _task(spec, str(record["task_id"]))
        _write_labelled_extraction(
            output,
            trajectory_id=str(record["trajectory_id"]),
            task_family=task.task_family,
            accepted=record["terminal"]["acceptance"] == "ACCEPTED",
        )

    corpus, exclusions = _corpus(spec, output, records)

    assert corpus is not None
    assert [item.trajectory_id for item in corpus.trajectories] == [
        "matrix-determinant-01-r01",
        "matrix-determinant-01-r02",
    ]
    assert exclusions == [
        {
            "trajectory_id": "integer-bezout-01-r02",
            "reason": "terminal verifier outcome is inconclusive",
        },
        {
            "trajectory_id": "integer-bezout-01-r01",
            "reason": "task group has only one labelled trajectory after exclusions",
        },
    ]


def test_only_singleton_labelled_groups_return_exclusions_without_corpus(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC_PATH)
    records = [
        {
            "trajectory_id": "integer-bezout-01-r01",
            "task_id": "integer-bezout-01",
            "terminal": {"acceptance": "ACCEPTED"},
        },
        {
            "trajectory_id": "matrix-determinant-01-r01",
            "task_id": "matrix-determinant-01",
            "terminal": {"acceptance": "REJECTED"},
        },
    ]
    for record in records:
        task = _task(spec, str(record["task_id"]))
        _write_labelled_extraction(
            tmp_path,
            trajectory_id=str(record["trajectory_id"]),
            task_family=task.task_family,
            accepted=record["terminal"]["acceptance"] == "ACCEPTED",
        )

    corpus, exclusions = _corpus(spec, tmp_path, records)

    assert corpus is None
    assert exclusions == [
        {
            "trajectory_id": "integer-bezout-01-r01",
            "reason": "task group has only one labelled trajectory after exclusions",
        },
        {
            "trajectory_id": "matrix-determinant-01-r01",
            "reason": "task group has only one labelled trajectory after exclusions",
        },
    ]


def test_committed_real_study_manifest_binds_every_artifact() -> None:
    manifest = json.loads((STUDY_PATH / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_revision"] == ("cd7e5d52abe3556a8ad0beb50cb82e9f4e42c86c")
    assert manifest["source_tree_clean_at_start"] is True
    assert manifest["codex"]["model"] == "gpt-5.4-mini"
    assert manifest["codex"]["reasoning_effort"] == "medium"
    assert manifest["training_performed"] is False
    assert manifest["scorer_intervention"] is False
    assert manifest["causal_claim_authorized"] is False
    assert len(manifest["artifacts"]) == 285
    assert {
        path.relative_to(STUDY_PATH).as_posix()
        for path in STUDY_PATH.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    } == set(manifest["artifacts"])
    assert all(
        file_digest(STUDY_PATH / relative) == expected
        for relative, expected in manifest["artifacts"].items()
    )


def test_committed_real_study_preserves_inconclusive_and_negative_result() -> None:
    summary = json.loads((STUDY_PATH / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((STUDY_PATH / "runs").glob("*/run.json"))
    ]

    assert summary["run_count"] == 16
    assert summary["accepted_count"] == 15
    assert summary["rejected_count"] == 0
    assert summary["labelled_trajectory_count"] == 15
    assert summary["excluded"] == [
        {
            "reason": "terminal verifier outcome is inconclusive",
            "trajectory_id": "polynomial-gcd-bezout-01-r03",
        }
    ]
    assert all(metric["brier_score"] == 0.0 for metric in summary["metrics"].values())
    assert all(
        metric["mean_absolute_error"] == 0.0 for metric in summary["metrics"].values()
    )
    assert sum(record["command"]["status"] == "TIMED_OUT" for record in records) == 1
    assert (
        sum(record["reasoning_protocol"]["status"] == "COMPLETE" for record in records)
        == 9
    )
    assert all(
        record["terminal"]["input_binding_valid"] is True
        and record["terminal"]["artifact_binding_valid"] is True
        for record in records
        if record["terminal"]["acceptance"] == "ACCEPTED"
    )
