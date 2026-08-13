from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import tools.benchmark_plan.compiler as planner
from tests.boundary.process.tooling.ci import ROOT

_MISSING = object()


def _load_adapter(module_state: pytest.MonkeyPatch) -> ModuleType:
    path = ROOT / ".github/scripts/plan-benchmarks"
    loader = SourceFileLoader("plan_benchmarks", str(path))
    spec = importlib.util.spec_from_loader("plan_benchmarks", loader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module_state.setitem(sys.modules, "plan_benchmarks", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_plan_benchmarks_adapter() -> Iterator[ModuleType]:
    previous = sys.modules.get("plan_benchmarks", _MISSING)
    with pytest.MonkeyPatch.context() as module_state:
        yield _load_adapter(module_state)
    assert sys.modules.get("plan_benchmarks", _MISSING) is previous


def test_loaded_planner_adapter_restores_module_state(
    isolated_plan_benchmarks_adapter: ModuleType,
) -> None:
    assert sys.modules["plan_benchmarks"] is isolated_plan_benchmarks_adapter
    assert isolated_plan_benchmarks_adapter.plan is planner.plan


def _patch_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = SimpleNamespace(
        id="mathematical-benchmarks-v1",
        tasks=(SimpleNamespace(path=Path("autoformalization-semantic-audit")),),
    )
    monkeypatch.setattr(
        planner,
        "_membership",
        lambda: (
            {"autoformalization-semantic-audit": [(suite.id, suite.tasks[0].path)]},
            {suite.id: suite},
        ),
    )
    monkeypatch.setattr(
        planner,
        "_topology_digest",
        lambda suites: "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(planner, "_digest", lambda path: "sha256:" + "b" * 64)


def test_task_documentation_does_not_select_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_plan(monkeypatch)

    plan = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "autoformalization-semantic-audit/README.md"
        ],
        base="a" * 40,
        head="b" * 40,
    )

    assert plan["run-benchmark-oracle"] == "false"
    assert plan["benchmark-oracle-scope"] == "none"
    assert "task documentation change" in plan["benchmark-plan-reasons"]


@pytest.mark.parametrize(
    "path",
    [".github/scripts/_ci_paths.py", "tools/check_benchmark_static.py"],
)
def test_benchmark_control_tools_run_contract_gate_without_oracle(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_plan(monkeypatch)

    plan = planner.plan([path], base="a" * 40, head="b" * 40)

    assert plan["run-benchmark-check"] == "true"
    assert plan["run-benchmark-record-schema"] == "true"
    assert plan["run-benchmark-oracle"] == "false"


def test_task_environment_selects_exact_task_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_plan(monkeypatch)

    plan = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "autoformalization-semantic-audit/environment/input.json"
        ],
        base="a" * 40,
        head="b" * 40,
    )

    assert plan["run-benchmark-oracle"] == "true"
    assert plan["benchmark-oracle-scope"] == "changed-tasks"
    assert "autoformalization-semantic-audit" in plan["benchmark-oracle-matrix"]


def test_shared_environment_profile_escalates_only_on_integration_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_plan(monkeypatch)

    pull_request = planner.plan(
        ["benchmarks/environment-profiles.toml"],
        event="pull_request",
        base="a" * 40,
        head="b" * 40,
    )
    merge_group = planner.plan(
        ["benchmarks/environment-profiles.toml"],
        event="merge_group",
        base="a" * 40,
        head="b" * 40,
    )

    assert pull_request["run-benchmark-oracle"] == "false"
    assert merge_group["run-benchmark-oracle"] == "true"
    assert merge_group["benchmark-oracle-scope"] == "all"


def test_main_push_is_an_integration_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_plan(monkeypatch)

    push = planner.plan(
        ["benchmarks/environment-profiles.toml"],
        event="push",
        base="a" * 40,
        head="b" * 40,
    )

    assert push["run-benchmark-oracle"] == "true"
    assert push["benchmark-oracle-scope"] == "all"
    assert push["run-benchmark-inventory"] == "true"
    assert push["benchmark-plan-mode"] == "integration"
