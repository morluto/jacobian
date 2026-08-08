from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from benchmarks.tooling.command_runner import ToolCommandResult, ToolCommandStatus
from benchmarks.tooling.trajectory_value_calibration import (
    HarborTaskContract,
    TrajectoryValueCalibrationSpec,
    _artifact_manifest,
    _codex_arguments,
    _run_one,
    _task_contract,
    load_spec,
    summarize,
)
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "benchmarks/config/trajectory-value-calibration-v1.json"
EXTENSION_SPEC = (
    ROOT / "benchmarks/config/trajectory-value-calibration-extension-v1.json"
)


def _record(
    dataset: str,
    task: str,
    acceptance: str,
    *,
    input_binding_valid: bool | None = None,
) -> dict[str, object]:
    terminal: dict[str, object] = {"acceptance": acceptance}
    if input_binding_valid is not None:
        terminal["input_binding_valid"] = input_binding_valid
    return {
        "dataset_id": dataset,
        "task_id": task,
        "terminal": terminal,
    }


def test_preregistered_spec_is_closed_and_binds_real_harbor_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration._harbor_task_digest",
        lambda _path: "sha256:" + "0" * 64,
    )
    spec = load_spec(SPEC)
    assert len(spec.candidates) == 8
    assert spec.repetitions_per_candidate == 2
    assert spec.retries_for_wrong_answers == 0
    contracts = [_task_contract(candidate) for candidate in spec.candidates]
    assert all(contract.harbor_digest.startswith("sha256:") for contract in contracts)
    assert all(
        "verifier.py" in contract.verifier_file_digests for contract in contracts
    )


def test_unknown_spec_fields_fail_closed() -> None:
    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["post_label_tuning"] = True
    with pytest.raises(ValidationError):
        TrajectoryValueCalibrationSpec.model_validate(value)


def test_extension_is_a_separate_preregistered_candidate_batch() -> None:
    initial = load_spec(SPEC)
    extension = load_spec(EXTENSION_SPEC)
    assert extension.calibration_id != initial.calibration_id
    assert len(extension.candidates) == 8
    assert not {
        (candidate.dataset_id, candidate.task_id) for candidate in initial.candidates
    }.intersection(
        (candidate.dataset_id, candidate.task_id) for candidate in extension.candidates
    )


def test_candidate_coverage_is_required() -> None:
    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["candidates"] = value["candidates"][:4]
    with pytest.raises(ValidationError, match="every declared trap"):
        TrajectoryValueCalibrationSpec.model_validate(value)


def test_selection_uses_only_labelled_terminal_outcomes_in_candidate_order() -> None:
    spec = load_spec(SPEC)
    records: list[dict[str, object]] = []
    expected: list[str] = []
    for index, candidate in enumerate(spec.candidates):
        if index < 5:
            outcomes = ("ACCEPTED", "REJECTED")
            if index < 4:
                expected.append(candidate.task_id)
        elif index == 5:
            outcomes = ("ACCEPTED", "ACCEPTED")
        elif index == 6:
            outcomes = ("REJECTED", "REJECTED")
        else:
            outcomes = ("ACCEPTED", "INCONCLUSIVE")
        records.extend(
            _record(candidate.dataset_id, candidate.task_id, outcome)
            for outcome in outcomes
        )
    summary = summarize(spec, records)
    assert [row["task_id"] for row in summary["selected_tasks"]] == expected
    assert summary["accepted"] == 8
    assert summary["rejected"] == 7
    assert summary["inconclusive"] == 1


def test_input_binding_failures_remain_inconclusive_in_summary() -> None:
    spec = load_spec(SPEC)
    candidate = spec.candidates[0]
    records = [
        _record(
            candidate.dataset_id,
            candidate.task_id,
            "REJECTED",
            input_binding_valid=False,
        ),
        _record(candidate.dataset_id, candidate.task_id, "ACCEPTED"),
    ]

    summary = summarize(spec, records)

    first = summary["candidate_results"][0]
    assert first["accepted"] == 1
    assert first["rejected"] == 0
    assert first["inconclusive"] == 1
    assert summary["accepted"] == 1
    assert summary["rejected"] == 0
    assert summary["inconclusive"] == 1


def test_calibration_codex_command_disables_web_search_under_ignored_config(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC)
    arguments = _codex_arguments(
        workspace=tmp_path,
        spec=spec,
        mcp_url="http://127.0.0.1:8765/mcp",
        prompt="calibration prompt",
    )

    assert "--ignore-user-config" in arguments
    assert 'web_search="disabled"' in arguments


def test_artifact_manifest_excludes_only_the_root_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output"
    nested = output / "runs" / "case" / "workspace" / "evidence"
    nested.mkdir(parents=True)
    (output / "manifest.json").write_text("root", encoding="utf-8")
    (nested / "manifest.json").write_text("nested", encoding="utf-8")
    (nested / "result.json").write_text("result", encoding="utf-8")

    manifest = _artifact_manifest(output)

    assert "manifest.json" not in manifest
    assert "runs/case/workspace/evidence/manifest.json" in manifest
    assert "runs/case/workspace/evidence/result.json" in manifest


def test_unsafe_calibration_workspace_records_inconclusive_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_spec(SPEC)
    candidate = spec.candidates[0]
    task_root = tmp_path / "task"
    (task_root / "environment").mkdir(parents=True)
    files = {
        "instruction.md": task_root / "instruction.md",
        "input.json": task_root / "environment" / "input.json",
        "submission_schema.json": task_root / "environment" / "submission_schema.json",
    }
    for name, path in files.items():
        path.write_text(name, encoding="utf-8")
    task = HarborTaskContract(
        dataset_id=candidate.dataset_id,
        task_id=candidate.task_id,
        path=task_root,
        harbor_digest="sha256:" + "0" * 64,
        public_file_digests={
            name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
        },
        verifier_file_digests={},
    )

    async def inspect_surface_fixture(_url: str, _timeout: int) -> dict[str, str]:
        return {"surface_digest": "sha256:" + "1" * 64}

    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration.inspect_surface",
        inspect_surface_fixture,
    )
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration.run_operator_command",
        lambda *args, **kwargs: ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=b"{}",
            stderr=b"",
        ),
    )
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration._reasoning_run_ids",
        lambda _path: ("run",),
    )
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration._required_reasoning_log",
        lambda _url, _run_ids: "",
    )
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration._verification_outcome",
        lambda **_kwargs: {
            "verifier_execution_status": "COMPLETED",
            "acceptance": "ACCEPTED",
            "reason": "TERMINAL_CLEAN_ROOM_REWARD",
            "input_binding_valid": True,
            "artifact_binding_valid": True,
            "reward": {"reward": 1.0},
        },
    )
    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration._copy_workspace",
        lambda _source, _destination: (_ for _ in ()).throw(
            RuntimeError("symlink is forbidden in evaluation evidence: workspace/link")
        ),
    )

    @contextmanager
    def mcp_server(**_kwargs):
        yield "http://127.0.0.1:8765/mcp"

    monkeypatch.setattr(
        "benchmarks.tooling.trajectory_value_calibration._mcp_server",
        mcp_server,
    )

    record = _run_one(
        spec=spec,
        candidate=candidate,
        task=task,
        repetition=1,
        output=tmp_path / "out",
    )

    assert record["terminal"]["acceptance"] == "INCONCLUSIVE"
    assert record["terminal"]["reason"] == "UNSAFE_WORKSPACE_EVIDENCE"
    assert record["terminal"]["artifact_binding_valid"] is False
    assert (tmp_path / "out" / "runs" / record["trajectory_id"] / "run.json").is_file()


def test_schema_matches_checked_in_contract() -> None:
    checked_in = json.loads(
        (
            ROOT
            / "docs/reference/evaluations/schemas/trajectory-value-calibration-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert checked_in == TrajectoryValueCalibrationSpec.model_json_schema()
