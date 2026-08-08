from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from benchmarks.tooling.command_runner import ToolCommandStatus
from benchmarks.tooling.trajectory_value_calibration import (
    CalibrationCandidate,
    HarborTaskContract,
    _task_contract,
)
from benchmarks.tooling.trajectory_value_hypothesis_study import (
    TrajectoryValueHypothesisStudySpec,
    _codex_arguments,
    _local_auth_status,
    _package_comparison,
    _publish_workspace_and_extract,
    _recover_interrupted_record,
    _verify_terminal,
    analyze_comparison,
    load_historical_corpus,
    load_hypothesis_spec,
    run_study,
)
from benchmarks.tooling.trajectory_value_mixed_contract import (
    FrozenMixedTask,
    ValidatedFrozenStudy,
)
from benchmarks.tooling.trajectory_value_study_verifier import file_digest
from pydantic import ValidationError
from tests.unit.tooling.test_trajectory_value_abstraction import _controlled_corpus

from jacobian.eval.trajectory_state import CleanRoomTerminalEvidence, TerminalAcceptance
from jacobian.eval.trajectory_value_abstraction import (
    evaluate_semantic_trajectories,
)

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "benchmarks/config/trajectory-value-hypothesis-study-v1.json"
SCHEMA = (
    ROOT
    / "docs/reference/evaluations/schemas/trajectory-value-hypothesis-study-v1.schema.json"
)
STUDY = ROOT / "benchmarks/studies/trajectory-value-hypothesis-codex-v1"
HISTORICAL_MIXED = STUDY / "frozen-contracts/trajectory-value-mixed-study-v1.json"


def _value() -> dict[str, object]:
    return cast(dict[str, object], json.loads(SPEC.read_text(encoding="utf-8")))


def _load_historical() -> tuple[
    TrajectoryValueHypothesisStudySpec, ValidatedFrozenStudy
]:
    return load_hypothesis_spec(
        SPEC,
        verify_current_tasks=False,
        historical_mixed_path=HISTORICAL_MIXED,
    )


def test_preregistration_binds_frozen_24_rollout_mixed_study() -> None:
    spec, validated = _load_historical()
    assert spec.analysis_id == "trajectory-value-hypothesis-codex-v1"
    assert [task.task_id for task in validated.contract.tasks] == [
        "graph-artifact-composition",
        "apollonius-gap-repair",
        "rp2-homology-lattice",
    ]
    assert len(validated.contract.tasks) * validated.contract.repetitions_per_task == 24
    assert spec.h3.threshold_tuned_on_main_labels is False
    assert spec.scorer_intervention is False
    assert spec.learned_components is False


def test_preregistration_is_closed_and_estimator_order_is_fixed() -> None:
    value = _value()
    value["post_label_threshold"] = -0.1
    with pytest.raises(ValidationError):
        TrajectoryValueHypothesisStudySpec.model_validate(value)

    value = _value()
    h1 = cast(dict[str, object], value["h1"])
    estimators = cast(list[str], h1["typed_estimators"])
    estimators.reverse()
    with pytest.raises(ValidationError, match="preregistered order"):
        TrajectoryValueHypothesisStudySpec.model_validate(value)


def test_mixed_study_digest_substitution_fails_closed(tmp_path: Path) -> None:
    value = _value()
    reference = cast(dict[str, object], value["mixed_study"])
    reference["file_digest"] = "sha256:" + "0" * 64
    altered = TrajectoryValueHypothesisStudySpec.model_validate(value)
    temporary = tmp_path / "invalid-hypothesis-study.json"
    temporary.write_text(altered.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="digest drift"):
        load_hypothesis_spec(
            temporary,
            verify_current_tasks=False,
            historical_mixed_path=HISTORICAL_MIXED,
        )


def test_current_contract_drift_cannot_substitute_or_authorize_execution() -> None:
    with pytest.raises(ValueError, match="frozen mixed-study file digest drift"):
        load_hypothesis_spec(SPEC)
    with pytest.raises(ValueError, match="cannot authorize new execution"):
        load_hypothesis_spec(SPEC, historical_mixed_path=HISTORICAL_MIXED)


def test_external_execution_requires_explicit_opt_in(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="without --execute"):
        run_study(SPEC, tmp_path / "result", execute=False)


def test_local_chatgpt_login_accepts_codex_stderr_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_hypothesis_study.run_operator_command",
        lambda *_args, **_kwargs: SimpleNamespace(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=b"",
            stderr=b"Logged in using ChatGPT\n",
        ),
    )
    assert _local_auth_status() == "Logged in using ChatGPT"


def test_codex_command_freezes_model_isolation_and_no_web_or_retry() -> None:
    _spec, validated = _load_historical()
    arguments = _codex_arguments(
        workspace=ROOT,
        mixed=validated.contract,
        mcp_url="http://127.0.0.1:8123/mcp",
        prompt=validated.contract.agent_instructions,
    )
    joined = " ".join(arguments)
    assert "gpt-5.4-mini" in arguments
    assert 'model_reasoning_effort="medium"' in arguments
    assert "--ephemeral" in arguments
    assert "--ignore-user-config" in arguments
    assert "--ignore-rules" in arguments
    assert "OPENAI_API_KEY" not in joined
    assert "retry" not in joined.lower()


def _first_task() -> tuple[FrozenMixedTask, HarborTaskContract]:
    _spec, validated = _load_historical()
    task = validated.contract.tasks[0]
    contract = _task_contract(
        CalibrationCandidate(
            dataset_id=task.dataset_id,
            task_id=task.task_id,
            task_family=task.task_family,
            calibration_tags=task.calibration_tags,
        )
    )
    return task, contract


def _workspace(tmp_path: Path, contract: HarborTaskContract) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name, relative in {
        "instruction.md": Path("instruction.md"),
        "input.json": Path("environment/input.json"),
        "submission_schema.json": Path("environment/submission_schema.json"),
    }.items():
        shutil.copyfile(contract.path / relative, workspace / name)
    (workspace / "submission.json").write_text("{}\n", encoding="utf-8")
    return workspace


def test_rejected_reward_remains_an_exact_bound_negative_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, contract = _first_task()
    workspace = _workspace(tmp_path, contract)
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_hypothesis_study.run_verifier_in_child",
        lambda **_kwargs: {
            "reward": 0.0,
            "correctness": 1.0,
            "evidence_validity": 0.0,
            "false_certification": False,
        },
    )
    outcome, terminal = _verify_terminal(
        task=task,
        contract=contract,
        workspace=workspace,
        run_dir=tmp_path / "run",
        source_binding_digest="sha256:" + "9" * 64,
        command_status=ToolCommandStatus.EXITED,
        exit_code=0,
    )
    assert outcome["acceptance"] == "REJECTED"
    assert outcome["submission_evidence_valid"] is False
    assert outcome["artifact_binding_valid"] is True
    assert terminal.acceptance is TerminalAcceptance.REJECTED
    assert terminal.input_binding_valid is True
    assert terminal.artifact_binding_valid is True


def test_public_input_drift_is_inconclusive_and_verifier_is_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, contract = _first_task()
    workspace = _workspace(tmp_path, contract)
    (workspace / "input.json").write_text("{}\n", encoding="utf-8")

    def forbidden(**_kwargs: object) -> None:
        raise AssertionError("verifier must not run after public input drift")

    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_hypothesis_study.run_verifier_in_child",
        forbidden,
    )
    outcome, terminal = _verify_terminal(
        task=task,
        contract=contract,
        workspace=workspace,
        run_dir=tmp_path / "run",
        source_binding_digest="sha256:" + "9" * 64,
        command_status=ToolCommandStatus.EXITED,
        exit_code=0,
    )
    assert outcome["reason"] == "PUBLIC_INPUT_DRIFT"
    assert terminal.acceptance is TerminalAcceptance.INCONCLUSIVE
    assert terminal.verifier_execution_status == "ERROR"


def test_interrupted_rollout_is_excluded_without_rerunning_model(
    tmp_path: Path,
) -> None:
    _spec, validated = _load_historical()
    task = validated.contract.tasks[1]
    run_dir = tmp_path / "runs" / "apollonius-gap-repair-main-r04"
    run_dir.mkdir(parents=True)
    fixture = ROOT / "tests/unit/tooling/fixtures/trajectory_state/pr1_gcd_real_codex"
    shutil.copyfile(fixture / "codex.jsonl", run_dir / "codex.jsonl")
    shutil.copyfile(fixture / "codex.stderr", run_dir / "codex.stderr")
    shutil.copyfile(fixture / "reasoning-log.jsonl", run_dir / "reasoning-log.jsonl")
    (run_dir / "surface.json").write_text(
        json.dumps(
            {
                "surface_digest": "sha256:" + "1" * 64,
                "catalog": {
                    "catalog_digest": "sha256:" + "2" * 64,
                    "policy_digest": "sha256:" + "3" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "verifier.json").write_text(
        json.dumps(
            {
                "verifier_digest": "sha256:" + "4" * 64,
                "verifier_execution_status": "COMPLETED",
                "acceptance": "ACCEPTED",
                "input_binding_valid": True,
                "artifact_binding_valid": True,
            }
        ),
        encoding="utf-8",
    )
    record = _recover_interrupted_record(
        task=task,
        repetition=4,
        run_dir=run_dir,
        prior_revision="5" * 40,
    )
    assert record["terminal"]["acceptance"] == "INCONCLUSIVE"
    assert record["rerun_performed"] is False
    assert record["exclusion_reason"].startswith("runner extraction failed")
    failure = json.loads(
        (run_dir / "infrastructure-failure.json").read_text(encoding="utf-8")
    )
    assert failure["disposition"] == "INCONCLUSIVE"
    assert failure["rerun_performed"] is False
    assert not (run_dir / "extraction.json").exists()


def test_live_extraction_failure_preserves_workspace_and_becomes_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _spec, validated = _load_historical()
    task = validated.contract.tasks[2]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "submission.json").write_text('{"answer": 1}\n', encoding="utf-8")
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "rp2-homology-lattice-main-r01"
    run_dir.mkdir(parents=True)
    terminal = CleanRoomTerminalEvidence(
        verifier_digest="sha256:" + "4" * 64,
        source_binding_digest=file_digest(transcript),
        clean_room=True,
        verifier_execution_status="COMPLETED",
        acceptance=TerminalAcceptance.REJECTED,
        input_binding_valid=True,
        artifact_binding_valid=True,
    )

    def fail_extraction(*_args: object, **_kwargs: object) -> None:
        raise ValueError("noncanonical candidate value")

    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_hypothesis_study.extract_codex_trajectory",
        fail_extraction,
    )
    analysis_terminal, reason = _publish_workspace_and_extract(
        transcript=transcript,
        workspace=workspace,
        run_dir=run_dir,
        task=task,
        source_revision="5" * 40,
        original_terminal=terminal,
        original_verifier={"acceptance": "REJECTED", "reward": {"reward": 0.0}},
    )
    assert analysis_terminal.acceptance is TerminalAcceptance.INCONCLUSIVE
    assert reason == "runner extraction failed after raw submission publication"
    assert (run_dir / "workspace/submission.json").read_text(encoding="utf-8") == (
        '{"answer": 1}\n'
    )
    failure = json.loads(
        (run_dir / "infrastructure-failure.json").read_text(encoding="utf-8")
    )
    assert failure["missing_artifacts"] == ["extraction"]
    assert failure["rerun_performed"] is False
    assert not (run_dir / "extraction.json").exists()


def test_comparison_packaging_is_deterministic_and_preserves_exact_bytes(
    tmp_path: Path,
) -> None:
    comparison = evaluate_semantic_trajectories(_controlled_corpus())
    raw = (comparison.model_dump_json(indent=2) + "\n").encode()
    (tmp_path / "comparison.json").write_bytes(raw)
    metadata = _package_comparison(tmp_path, source_revision="5" * 40)
    compressed = (tmp_path / "comparison.json.gz").read_bytes()
    assert gzip.decompress(compressed) == raw
    assert metadata["uncompressed_file_digest"] == (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    assert metadata["mtime"] == 0
    assert not (tmp_path / "comparison.json").exists()


def test_controlled_comparison_exercises_preregistered_hypothesis_analysis() -> None:
    source = evaluate_semantic_trajectories(_controlled_corpus())
    value = _value()
    value["analysis_id"] = source.corpus_id
    spec = TrajectoryValueHypothesisStudySpec.model_validate(value)
    result = analyze_comparison(
        spec, source, run_count=len(source.source_corpus.trajectories)
    )
    assert result["mixed_terminal_outcomes"] is True
    assert result["h1"]["h1_directionally_supported"] is True
    assert result["h2"]["mixed_outcome_compatible_pair_count"] > 0
    assert result["h2"]["h2_directionally_supported"] is True
    assert result["h3"]["threshold_tuned_on_main_labels"] is False
    assert result["scorer_intervention"] is False
    assert result["causal_claim_authorized"] is False


def test_schema_matches_authoritative_preregistration_contract() -> None:
    assert json.loads(SCHEMA.read_text(encoding="utf-8")) == (
        TrajectoryValueHypothesisStudySpec.model_json_schema()
    )


def test_committed_hypothesis_study_manifest_binds_every_artifact() -> None:
    manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {
        path.relative_to(STUDY).as_posix()
        for path in STUDY.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert artifact_paths == set(manifest["artifacts"])
    assert all(
        file_digest(STUDY / relative) == expected
        for relative, expected in manifest["artifacts"].items()
    )
    packaging = manifest["packaging"]["comparison"]
    compressed = (STUDY / packaging["path"]).read_bytes()
    raw_comparison = gzip.decompress(compressed)
    assert packaging["encoding"] == "gzip"
    assert packaging["mtime"] == 0
    assert packaging["source_revision"] == ("b3b00a39b9efaec9991747526b85ab2e2eaf2105")
    assert packaging["uncompressed_bytes"] == len(raw_comparison)
    assert packaging["uncompressed_file_digest"] == (
        "sha256:" + hashlib.sha256(raw_comparison).hexdigest()
    )
    assert (
        packaging["compressed_file_digest"] == manifest["artifacts"][packaging["path"]]
    )
    attempt_ids = [
        trajectory_id
        for attempt in manifest["execution_attempts"]
        for trajectory_id in attempt["trajectory_ids"]
    ]
    assert len(attempt_ids) == len(set(attempt_ids)) == 24
    assert [
        attempt["completed_normally"] for attempt in manifest["execution_attempts"]
    ] == [
        False,
        False,
        True,
    ]
    assert manifest["codex"]["authentication"] == "Logged in using ChatGPT"
    assert manifest["codex"]["api_key_environment_forwarded"] is False
    assert manifest["budgets"]["retries_for_wrong_answers"] == 0
    reanalysis = manifest["posthoc_reanalysis"]
    assert reanalysis["source_revision"] == packaging["source_revision"]
    assert reanalysis["model_rerun"] is False
    assert reanalysis["terminal_verifier_outcomes_changed"] is False
    assert reanalysis["previous"]["comparison_packaging"]["source_revision"] == (
        "4c09fbac6eefd3521d3ec75cd8d4b53c9e68498c"
    )


def test_historical_corpus_and_analysis_match_frozen_outcomes() -> None:
    _spec, _validated = _load_historical()
    corpus = load_historical_corpus(STUDY / "corpus.json")
    analysis = json.loads((STUDY / "analysis.json").read_text(encoding="utf-8"))
    assert len(corpus.trajectories) == 19
    assert (analysis["accepted_count"], analysis["rejected_count"]) == (9, 10)
    assert analysis["h1"]["h1_directionally_supported"] is False
    assert analysis["h2"]["h2_directionally_supported"] is False
    assert analysis["h3"]["h3_directionally_supported"] is True
    assert analysis["h3"]["supported_estimators"] == ["REASONING_TEXT"]
    assert analysis["h2"]["mixed_outcome_compatible_pair_count"] == 30
    assert analysis["h2"]["hybrid_separated_pair_count"] == 22


def test_historical_corpus_rejects_substituted_soft_state_binding(
    tmp_path: Path,
) -> None:
    payload = json.loads((STUDY / "corpus.json").read_text(encoding="utf-8"))
    payload["trajectories"][0]["extraction"]["states"][0]["soft_state_digest"] = (
        "sha256:" + "0" * 64
    )
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"artifacts": {"corpus.json": file_digest(corpus_path)}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="historical soft-state binding drift"):
        load_historical_corpus(corpus_path)


def test_committed_hypothesis_labels_are_clean_room_and_fail_closed() -> None:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((STUDY / "runs").glob("*/run.json"))
    ]
    verifiers = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((STUDY / "runs").glob("*/verifier.json"))
    ]
    assert len(records) == len(verifiers) == 24
    assert (
        sum(record["terminal"]["acceptance"] == "ACCEPTED" for record in records) == 11
    )
    assert (
        sum(record["terminal"]["acceptance"] == "REJECTED" for record in records) == 11
    )
    assert (
        sum(record["terminal"]["acceptance"] == "INCONCLUSIVE" for record in records)
        == 2
    )
    assert all(verifier["clean_room"] is True for verifier in verifiers)
    assert all(verifier["false_certification"] is False for verifier in verifiers)
    excluded_failures = [
        record for record in records if record.get("rerun_performed") is False
    ]
    assert [record["trajectory_id"] for record in excluded_failures] == [
        "apollonius-gap-repair-main-r04",
        "rp2-homology-lattice-main-r01",
    ]
    assert all(
        record["terminal"]["acceptance"] == "INCONCLUSIVE"
        for record in excluded_failures
    )
