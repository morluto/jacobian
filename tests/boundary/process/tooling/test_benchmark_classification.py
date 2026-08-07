from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from tests.boundary.process.tooling.ci import ROOT

_MISSING = object()


def _load(module_state: pytest.MonkeyPatch) -> ModuleType:
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
def isolated_plan_benchmarks_module() -> Iterator[ModuleType]:
    previous = sys.modules.get("plan_benchmarks", _MISSING)
    with pytest.MonkeyPatch.context() as module_state:
        yield _load(module_state)
    assert sys.modules.get("plan_benchmarks", _MISSING) is previous


def test_loaded_planner_restores_module_state(
    isolated_plan_benchmarks_module: ModuleType,
) -> None:
    assert sys.modules["plan_benchmarks"] is isolated_plan_benchmarks_module


def _patch_plan(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = SimpleNamespace(
        id="mathematical-benchmarks-v1",
        tasks=(SimpleNamespace(path=Path("autoformalization-semantic-audit")),),
    )
    monkeypatch.setattr(
        module,
        "_membership",
        lambda: (
            {"autoformalization-semantic-audit": [(suite.id, suite.tasks[0].path)]},
            {suite.id: suite},
        ),
    )
    monkeypatch.setattr(module, "_topology_digest", lambda suites: "sha256:" + "a" * 64)
    monkeypatch.setattr(module, "_digest", lambda path: "sha256:" + "b" * 64)


def test_task_documentation_does_not_select_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(monkeypatch)
    _patch_plan(module, monkeypatch)

    plan = module.plan(
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
    module = _load(monkeypatch)
    _patch_plan(module, monkeypatch)

    plan = module.plan([path], base="a" * 40, head="b" * 40)

    assert plan["run-benchmark-check"] == "true"
    assert plan["run-benchmark-record-schema"] == "true"
    assert plan["run-benchmark-oracle"] == "false"


def test_task_environment_selects_exact_task_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(monkeypatch)
    _patch_plan(module, monkeypatch)

    plan = module.plan(
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
    module = _load(monkeypatch)
    _patch_plan(module, monkeypatch)

    pull_request = module.plan(
        ["benchmarks/environment-profiles.toml"],
        event="pull_request",
        base="a" * 40,
        head="b" * 40,
    )
    merge_group = module.plan(
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
    module = _load(monkeypatch)
    _patch_plan(module, monkeypatch)

    push = module.plan(
        ["benchmarks/environment-profiles.toml"],
        event="push",
        base="a" * 40,
        head="b" * 40,
    )

    assert push["run-benchmark-oracle"] == "true"
    assert push["benchmark-oracle-scope"] == "all"
    assert push["run-benchmark-inventory"] == "true"
    assert push["benchmark-plan-mode"] == "integration"
