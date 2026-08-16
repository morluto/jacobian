from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.tooling import benchmark_contracts, benchmark_inventory
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import (
    load_registry,
    validate_global_task_ids,
)


def test_every_committed_benchmark_contract_is_valid() -> None:
    assert benchmark_contracts.validate_all() == []


def test_generated_benchmark_results_remain_ignored() -> None:
    ignored = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    assert "benchmarks/results/" in ignored


def test_registry_rejects_global_task_id_collisions() -> None:
    first, second, *_rest = load_registry()
    colliding = replace(second, tasks=(first.tasks[0],))

    with pytest.raises(HarborSuiteError, match="global task id"):
        validate_global_task_ids([first, colliding])


def test_inventory_covers_every_registered_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark_inventory, "task_digest", lambda _path: "a" * 64)
    inventory = benchmark_inventory.build_inventory()
    suites = load_registry()

    assert inventory["schema_version"] == "2"
    assert inventory["dataset_count"] == len(suites)
    assert inventory["task_count"] == sum(len(suite.tasks) for suite in suites)
    rendered = json.dumps(inventory)
    assert "verifier_digest" in rendered
    assert "verifier_support_digest" in rendered
    assert "submission_schema_digest" in rendered


def test_visible_submission_contracts_expose_only_result_and_witness() -> None:
    retired_contract_fields = {
        "allowed_assurance",
        "allowed_completeness",
        "assurance_ceiling",
        "conclusion",
        "scope",
        "verification_record",
        "limitations",
    }
    for suite in load_registry():
        for task in suite.tasks:
            schema = json.loads(
                (task.path / "environment" / "submission_schema.json").read_text()
            )
            properties = schema["properties"]
            assert set(properties) <= {"result", "witness"}, task.path
            assert "result" in schema["required"], task.path
            witness = properties.get("witness")
            if witness is not None:
                assert witness.get("minItems") in {0, 1}, task.path
                assert witness.get("maxItems") in {0, 1}, task.path
                assert witness.get("minItems") <= witness.get("maxItems"), task.path
            contract_path = task.path / "tests" / "public_contract.json"
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text())
                assert not (set(contract) & retired_contract_fields), task.path


def test_task_gap_records_preserve_only_historical_provenance() -> None:
    paths = sorted(
        Path("benchmarks/datasets/mathematical-benchmarks-v1").glob(
            "*/analysis/gap.json"
        )
    )
    assert paths, "expected historical gap records under mathematical-benchmarks-v1"
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["provenance_status"] == "historical"
        assert record["historical_provenance_id"].endswith(".operation-gap-analysis")
        assert "ledger_id" not in record


def test_regular_file_inside_rejects_symlinked_witness(tmp_path: Path) -> None:
    root = tmp_path / "solution"
    nested = root / "evidence"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (nested / "witness.bin").symlink_to(outside)
    assert (
        benchmark_contracts._regular_file_inside(root, "evidence/witness.bin") is None
    )
    regular = nested / "ok.bin"
    regular.write_bytes(b"ok")
    assert benchmark_contracts._regular_file_inside(root, "evidence/ok.bin") == regular


def test_regular_file_inside_rejects_symlinked_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    nested = outside / "evidence"
    nested.mkdir(parents=True)
    (nested / "ok.bin").write_bytes(b"ok")
    root = tmp_path / "solution"
    root.symlink_to(outside)
    assert benchmark_contracts._regular_file_inside(root, "evidence/ok.bin") is None
