"""Honesty checks for the frozen direct-MCP migration evaluation."""

from __future__ import annotations

import asyncio
import runpy
from pathlib import Path

import pytest
from benchmarks.tooling.codex_visibility import (
    _ROOT as VISIBILITY_ROOT,
)
from benchmarks.tooling.codex_visibility import (
    AdoptionExpectation,
    CueLevel,
    SurfaceArm,
    VisibilityCase,
    _normalized_skill_source,
    _visible_tool_names,
    classify_visibility,
)
from benchmarks.tooling.codex_visibility import load_suite as load_agent_suite
from benchmarks.tooling.mcp_catalog_evaluation import (
    TaskCategory,
    load_suite,
    run_evaluation,
)

from jacobian.catalog.builtins import BUILTIN_TOOLS

_CONFIG = Path(__file__).parents[1] / "config"
_LOCAL_SUITE = _CONFIG / "direct-mcp-catalog-evaluation-v1.json"
_AGENT_SUITE = _CONFIG / "direct-mcp-agent-adoption-v1.json"


def test_visibility_paths_are_repository_relative() -> None:
    assert Path(__file__).resolve().parents[2] == VISIBILITY_ROOT
    assert (VISIBILITY_ROOT / "benchmarks/tooling/codex_telemetry.py").is_file()


def test_frozen_corpus_covers_every_required_family_and_public_operation() -> None:
    suite = load_suite(_LOCAL_SUITE)
    public_ids = {tool.operation_id for tool in BUILTIN_TOOLS}

    assert {task.category for task in suite.tasks} == set(TaskCategory)
    assert {
        step.operation_id for task in suite.tasks for step in task.steps
    } <= public_ids
    assert {
        operation_id
        for task in suite.tasks
        for probe in task.discovery_probes
        for operation_id in probe.required_operation_ids
    } <= public_ids
    assert {
        operation_id
        for case in suite.semantic_discovery_cases
        for operation_id in case.required_operation_ids
    } <= public_ids


def test_agent_and_local_corpora_freeze_the_same_cases_and_operations() -> None:
    local = load_suite(_LOCAL_SUITE)
    agent = load_agent_suite(_AGENT_SUITE)
    local_operations = {
        task.case_id: tuple(step.operation_id for step in task.steps)
        for task in local.tasks
    }
    local_operations.update(
        {
            case.case_id: case.required_operation_ids
            for case in local.semantic_discovery_cases
        }
    )

    assert agent.schema_version == "3"
    assert {case.case_id for case in agent.cases} == set(local_operations)
    assert {
        case.case_id: case.expected_operation_ids for case in agent.cases
    } == local_operations


def test_surface_arms_are_disjoint_where_the_comparison_requires_it() -> None:
    all_names = {
        "math.find",
        "math.run",
        "matrix.determinant.compute",
        "sat.assignment.check",
    }

    assert _visible_tool_names(all_names, SurfaceArm.LEGACY) == {
        "math.find",
        "math.run",
    }
    assert _visible_tool_names(all_names, SurfaceArm.DIRECT) == {
        "matrix.determinant.compute",
        "sat.assignment.check",
    }
    assert _visible_tool_names(all_names, SurfaceArm.DIRECT_FIND) == all_names - {
        "math.run"
    }
    assert _visible_tool_names(all_names, SurfaceArm.FIND_ONLY) == {"math.find"}


def test_visibility_package_has_an_executable_module_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_main() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("benchmarks.tooling.codex_visibility.main", fake_main)
    runpy.run_module(
        "benchmarks.tooling.codex_visibility.__main__", run_name="__main__"
    )

    assert called is True


def test_isolated_skill_paths_normalize_across_filesystem_symlinks(
    tmp_path: Path,
) -> None:
    real_codex_home = tmp_path / "real-codex-home"
    real_codex_home.mkdir()
    skill = real_codex_home / "skills/.system/example/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("example", encoding="utf-8")
    alias = tmp_path / "alias-codex-home"
    alias.symlink_to(real_codex_home, target_is_directory=True)

    normalized, external = _normalized_skill_source(
        str(alias / "skills/.system/example/SKILL.md"),
        workspace=tmp_path,
        environment={"CODEX_HOME": str(real_codex_home), "HOME": str(tmp_path)},
    )

    assert normalized == "$CODEX_HOME/skills/.system/example/SKILL.md"
    assert external is False


def test_semantic_discovery_is_scored_without_requiring_execution() -> None:
    case = VisibilityCase(
        case_id="sandpile-vocabulary",
        category="SEMANTIC_DISCOVERY",
        cue_level=CueLevel.AFFORDANCE,
        prompt="Find the installed sandpile vocabulary.",
        expectation=AdoptionExpectation.DISCOVER,
        expected_operation_ids=("graph.chip_firing.critical_group.compute",),
    )

    classification = classify_visibility(
        case,
        {
            "operation_descriptions": [
                {
                    "match_ids": ["graph.chip_firing.critical_group.compute"],
                    "operation_id": None,
                }
            ],
            "operation_describe_index_calls": 1,
            "mcp_calls": ["math.find"],
        },
    )

    assert classification["contract_satisfied"] is True
    assert classification["observed"]["invoked"] is False


def test_local_catalog_controls_pass_but_external_removal_gates_remain_open() -> None:
    report = asyncio.run(run_evaluation(load_suite(_LOCAL_SUITE), list_repetitions=1))

    assert report["surface"]["complete_catalog_schema_coverage"] is True
    assert report["summary"]["direct_task_success_count"] == 5
    assert report["summary"]["legacy_task_success_count"] == 5
    assert report["summary"]["exact_parity_task_count"] == 5
    assert report["summary"]["composition_success_count"] == 1
    assert report["summary"]["semantic_discovery_success_count"] == 2
    assert report["decision"]["math_run"]["removal_supported"] is False
    assert report["environment"]["evaluator_sha256"].startswith("sha256:")
    assert (
        report["decision"]["math_run"]["gates"]["deferred_client_discovery_observed"]
        == "UNMEASURED"
    )
    assert (
        report["decision"]["math_find"]["recommendation"]
        == "RETAIN_AS_SEMANTIC_DISCOVERY_ONLY"
    )
