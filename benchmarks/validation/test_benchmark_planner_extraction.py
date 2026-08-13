"""Regressions specific to package-owned benchmark planner extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import tools.benchmark_plan.compiler as planner
from tools.benchmark_plan.validation import validate_plan


@pytest.fixture(autouse=True)
def stable_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planner,
        "_digest",
        lambda path: f"sha256:{hashlib.sha256(path.name.encode()).hexdigest()}",
    )


def _host_matrix(result: dict[str, str]) -> list[dict[str, object]]:
    matrix = json.loads(result["benchmark-host-validation-matrix"])
    return [
        {key: value for key, value in entry.items() if key != "predicted_seconds"}
        for entry in matrix
    ]


def test_planner_digest_binds_every_declared_semantic_source() -> None:
    payload = "\n".join(
        f"{path.relative_to(planner.ROOT).as_posix()}\t{path.read_bytes().hex()}"
        for path in planner.PLANNER_DIGEST_SOURCES
    ).encode()
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()

    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py"
        ],
        event="pull_request",
    )

    assert result["benchmark-planner-digest"] == expected
    validate_plan(result)


def test_extracted_path_policy_change_runs_benchmark_contracts() -> None:
    result = planner.plan(
        ["tools/benchmark_plan/paths.py"],
        event="pull_request",
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["benchmark-plan-mode"] == "changed"
    validate_plan(result)


def test_benchmark_contract_tool_selects_its_owned_host_contract() -> None:
    path = "benchmarks/tooling/benchmark_contracts.py"

    result = planner.plan([path], event="pull_request")

    assert _host_matrix(result) == [
        {
            "name": "control-test_benchmark_contracts",
            "selector": "benchmarks/validation/test_benchmark_contracts.py",
            "keyword": "",
            "splits": 0,
            "group": 0,
        }
    ]
    validate_plan(result)


def test_path_adapter_source_is_part_of_the_declared_digest_contract() -> None:
    expected = planner.ROOT / "tools" / "benchmark_plan" / "paths.py"

    assert expected in planner.PLANNER_DIGEST_SOURCES
    assert Path("tools/benchmark_plan/paths.py") in {
        path.relative_to(planner.ROOT) for path in planner.PLANNER_DIGEST_SOURCES
    }
