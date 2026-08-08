"""Behavioral tests for timing-aware host-validation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling import command_runner
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.host_validation import (
    ExecutionProvenance,
    ShardResult,
    execute,
    pytest_arguments,
    verify_execution_sha,
    worker_allocation,
)
from benchmarks.tooling.validation_plan import full_host_validation

DIGEST = "sha256:" + "a" * 64
PROVENANCE = ExecutionProvenance(
    plan_head_sha="1" * 40,
    execution_sha="2" * 40,
    planner_digest=DIGEST,
    topology_digest=DIGEST,
    plan_digest=DIGEST,
    plan_receipt_digest=DIGEST,
)


def test_worker_allocation_caps_nested_parallelism() -> None:
    assert worker_allocation(entry_count=4, total_worker_budget=4, max_parallel=4) == (
        4,
        1,
    )
    assert worker_allocation(entry_count=1, total_worker_budget=8, max_parallel=4) == (
        1,
        2,
    )
    assert worker_allocation(entry_count=2, total_worker_budget=8, max_parallel=4) == (
        2,
        2,
    )
    assert worker_allocation(entry_count=4, total_worker_budget=2, max_parallel=4) == (
        2,
        1,
    )


def test_execution_sha_must_match_checked_out_merge_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(command_runner, "git_head_sha", lambda _root: "2" * 40)

    assert verify_execution_sha(tmp_path, "2" * 40) == "2" * 40
    with pytest.raises(HarborSuiteError, match="checked-out Git revision"):
        verify_execution_sha(tmp_path, "3" * 40)


def test_full_shard_uses_ci_timing_shape_and_least_duration(tmp_path: Path) -> None:
    entry = full_host_validation()[1]

    arguments = pytest_arguments(
        entry,
        timing_path=tmp_path / "benchmark-test-durations.json",
        workers=2,
        store_durations=False,
    )

    assert arguments[:4] == ("-n", "2", "--durations=10", "benchmarks/validation")
    assert arguments[arguments.index("--splits") + 1] == "4"
    assert arguments[arguments.index("--group") + 1] == "2"
    assert arguments[arguments.index("--splitting-algorithm") + 1] == "least_duration"
    assert "--store-durations" not in arguments


def test_all_shards_receive_same_deterministic_seed(tmp_path: Path) -> None:
    """Every full-host shard must get the same pytest-randomly seed so that
    pytest-split sees an identical collection order across CI runners."""

    entries = full_host_validation()
    seeds: set[str] = set()
    for entry in entries:
        arguments = pytest_arguments(
            entry,
            timing_path=tmp_path / "timings.json",
            workers=2,
            store_durations=False,
        )
        seed_index = arguments.index("--randomly-seed")
        seeds.add(arguments[seed_index + 1])
    assert len(seeds) == 1
    assert seeds.pop() == "0"


def test_keyword_filtered_entry_also_receives_deterministic_seed(
    tmp_path: Path,
) -> None:
    from benchmarks.tooling.validation_plan import task_host_validation

    entries = task_host_validation(
        Path(__file__).resolve().parents[2],
        "mathematical-benchmarks-v1",
        "algebraic-independence-transfer-audit",
    )
    assert entries
    for entry in entries:
        arguments = pytest_arguments(
            entry,
            timing_path=tmp_path / "timings.json",
            workers=1,
            store_durations=False,
        )
        assert "--randomly-seed" in arguments
        seed_index = arguments.index("--randomly-seed")
        assert arguments[seed_index + 1] == "0"


def test_local_full_run_uses_empty_timing_fallback_and_writes_bound_receipts(
    tmp_path: Path,
) -> None:
    entries = full_host_validation()
    observed: list[tuple[str, tuple[str, ...], int]] = []

    def fake_runner(entry, arguments, workers):
        observed.append((entry.name, arguments, workers))
        return ShardResult(status="EXITED", exit_code=0, actual_seconds=2.5)

    result = execute(
        entries,
        root=tmp_path,
        timing_path=tmp_path / ".ci/benchmark-test-durations.json",
        receipt_dir=tmp_path / "receipts",
        provenance=PROVENANCE,
        total_worker_budget=4,
        max_parallel=4,
        runner=fake_runner,
    )

    assert result == 0
    assert {name for name, _, _ in observed} == {entry.name for entry in entries}
    assert {workers for _, _, workers in observed} == {1}
    receipts = sorted((tmp_path / "receipts").rglob("pytest-receipt.json"))
    assert len(receipts) == 4
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["plan_head_sha"] == PROVENANCE.plan_head_sha
    assert payload["execution_sha"] == PROVENANCE.execution_sha
    assert payload["timing_digest"].startswith("sha256:")
    assert payload["total_worker_budget"] == 4
    assert payload["max_parallel"] == 4
    assert not (tmp_path / ".pytest_cache/benchmark-host").exists()


def test_failed_shard_fails_the_aggregate_exit_code(tmp_path: Path) -> None:
    entries = full_host_validation()[:2]

    def fake_runner(entry, _arguments, _workers):
        return ShardResult(
            status="EXITED",
            exit_code=1 if entry == entries[0] else 0,
            actual_seconds=1.0,
        )

    result = execute(
        entries,
        root=tmp_path,
        timing_path=tmp_path / "missing.json",
        receipt_dir=tmp_path / "receipts",
        provenance=PROVENANCE,
        total_worker_budget=2,
        max_parallel=2,
        runner=fake_runner,
    )

    assert result == 1


def test_invalid_timing_history_falls_back_without_mutating_input(
    tmp_path: Path,
) -> None:
    entry = full_host_validation()[0]
    timing_path = tmp_path / "durations.json"
    timing_path.write_text('{"bad": "duration"}\n', encoding="utf-8")
    observed: list[tuple[str, ...]] = []

    def fake_runner(_entry, arguments, _workers):
        observed.append(arguments)
        return ShardResult(status="EXITED", exit_code=0, actual_seconds=1.0)

    assert (
        execute(
            (entry,),
            root=tmp_path,
            timing_path=timing_path,
            receipt_dir=tmp_path / "receipts",
            provenance=PROVENANCE,
            total_worker_budget=1,
            max_parallel=1,
            runner=fake_runner,
        )
        == 0
    )
    assert timing_path.read_text(encoding="utf-8") == '{"bad": "duration"}\n'
    assert not (tmp_path / ".pytest_cache/benchmark-host").exists()
    assert observed


def test_duration_collection_writes_separate_output(tmp_path: Path) -> None:
    entry = full_host_validation()[0]
    timing_path = tmp_path / "input.json"
    timing_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output.json"

    def fake_runner(_entry, arguments, _workers):
        duration_path = Path(arguments[arguments.index("--durations-path") + 1])
        assert duration_path != timing_path
        duration_path.write_text(
            '{"benchmarks/validation/test_example.py::test_it": 1.0}\n',
            encoding="utf-8",
        )
        return ShardResult(status="EXITED", exit_code=0, actual_seconds=1.0)

    assert (
        execute(
            (entry,),
            root=tmp_path,
            timing_path=timing_path,
            receipt_dir=tmp_path / "receipts",
            provenance=PROVENANCE,
            total_worker_budget=4,
            max_parallel=4,
            store_durations=True,
            duration_output=output,
            runner=fake_runner,
        )
        == 0
    )
    assert timing_path.read_text(encoding="utf-8") == "{}\n"
    assert "test_example.py::test_it" in output.read_text(encoding="utf-8")
    assert not (tmp_path / ".pytest_cache/benchmark-host-timing").exists()


def test_conjecture_dataset_host_validation_discovers_all_dedicated_tests() -> None:
    """Dataset-wide conjecture-probes-v1 changes select every dedicated test file."""

    from benchmarks.tooling.validation_plan import dataset_host_validation

    entries = dataset_host_validation("conjecture-probes-v1")
    selectors = [entry.selector for entry in entries]

    project_root = Path(__file__).resolve().parents[2]
    expected_dir = project_root / "benchmarks" / "validation" / "conjecture_probes_v1"
    expected = sorted(
        str(p.relative_to(project_root).as_posix())
        for p in expected_dir.glob("test_*.py")
    )

    assert selectors == expected
    assert len(selectors) > 3


def test_non_conjecture_dataset_host_validation_uses_static_entries() -> None:
    """Other datasets still use the hand-maintained static file list."""

    from benchmarks.tooling.validation_plan import dataset_host_validation

    entries = dataset_host_validation("symbolic-coordination-v1")
    assert len(entries) == 1
    assert entries[0].selector == (
        "benchmarks/validation/symbolic_coordination_v1/test_pilot_contract.py"
    )
