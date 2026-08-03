from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.tooling import benchmark_contracts, benchmark_inventory
from benchmarks.tooling.harbor_suite import (
    HarborSuiteError,
    load_registry,
    validate_global_task_ids,
)

ASSURANCE_ORDER = ("UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED")


def test_every_committed_benchmark_contract_is_valid() -> None:
    assert benchmark_contracts.validate_all() == []


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

    assert inventory["schema_version"] == "1"
    assert inventory["dataset_count"] == len(suites)
    assert inventory["task_count"] == sum(len(suite.tasks) for suite in suites)
    rendered = json.dumps(inventory)
    assert "verifier_digest" in rendered
    assert "submission_schema_digest" in rendered


def test_visible_submission_contracts_match_evidence_and_assurance_limits() -> None:
    for suite in load_registry():
        for task in suite.tasks:
            config = tomllib.loads((task.path / "task.toml").read_text())
            ceiling = config["metadata"]["assurance_ceiling"]
            schema = json.loads(
                (task.path / "environment" / "submission_schema.json").read_text()
            )
            properties = schema["properties"]
            evidence = properties["evidence"]
            assert evidence.get("minItems") == 1, task.path
            assert evidence.get("maxItems") == 1, task.path

            assurance = properties["claimed_assurance"]
            advertised = assurance.get("enum", [assurance.get("const")])
            ceiling_index = ASSURANCE_ORDER.index(ceiling)
            assert ceiling in advertised, task.path
            assert all(
                value in ASSURANCE_ORDER[: ceiling_index + 1] for value in advertised
            ), task.path


def test_task_gap_records_preserve_only_historical_provenance() -> None:
    paths = sorted(
        Path("benchmarks/datasets/agent-workflow-v1").glob("*/analysis/gap.json")
    )
    assert paths, "expected historical gap records under agent-workflow-v1"
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["provenance_status"] == "historical"
        assert record["historical_provenance_id"].endswith(".capability-gap-analysis")
        assert "ledger_id" not in record
