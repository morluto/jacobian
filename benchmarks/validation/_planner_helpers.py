"""Shared helpers for Harbor planner validation suites."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import tools.benchmark_plan.compiler as planner
from tools.benchmark_plan.compiler import PLANNER_DIGEST_SOURCES

__all__ = [
    "PATH_POLICY_PATH",
    "PLANNER_DIGEST_SOURCES",
    "PLANNER_PATH",
    "ROOT",
    "VALIDATION_PLAN_PATH",
    "VALIDATOR_PATH",
    "ModuleType",
    "Path",
    "SimpleNamespace",
    "_assert_plan_valid",
    "_build_temp_topology",
    "_host_matrix",
    "_lane",
    "_load_script",
    "_matrix",
    "_matrix_tasks",
    "_raw_host_matrix",
    "hashlib",
    "json",
    "planner",
    "pytest",
    "sys",
]

ROOT = Path(__file__).resolve().parents[2]
PLANNER_PATH = ROOT / ".github" / "scripts" / "plan-benchmarks"
PATH_POLICY_PATH = ROOT / ".github" / "scripts" / "_ci_paths.py"
VALIDATION_PLAN_PATH = ROOT / "benchmarks" / "tooling" / "validation_plan.py"
VALIDATOR_PATH = ROOT / ".github" / "scripts" / "validate-benchmark-plan"


def _load_script(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_loader(
        module_name, SourceFileLoader(module_name, str(path))
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as module_state:
        module_state.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
    return module


def _matrix(result: dict[str, str]) -> list[dict[str, object]]:
    return json.loads(result["benchmark-oracle-matrix"])


def _matrix_tasks(result: dict[str, str]) -> set[str]:
    return {str(task) for item in _matrix(result) for task in item["tasks"]}


def _host_matrix(result: dict[str, str]) -> list[dict[str, object]]:
    matrix = json.loads(result["benchmark-host-validation-matrix"])
    return [
        {key: value for key, value in entry.items() if key != "predicted_seconds"}
        for entry in matrix
    ]


def _raw_host_matrix(result: dict[str, str]) -> list[dict[str, object]]:
    return json.loads(result["benchmark-host-validation-matrix"])


def _assert_plan_valid(result: dict[str, str]) -> None:
    payload = "\n".join(f"{key}={value}" for key, value in result.items()) + "\n"
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def _lane(result: dict[str, str], key: str) -> bool:
    return result[key] == "true"


def _build_temp_topology(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    """Build a minimal temp benchmark tree with one suite and one task.

    Returns the suite object whose ``path`` and ``suite_manifest`` point at
    real temp files so ``_topology_digest`` binds their content.
    """

    bench = tmp_path / "benchmarks"
    dataset_dir = bench / "datasets" / "alpha-v1"
    members_dir = dataset_dir / "members"
    members_dir.mkdir(parents=True)
    (bench / "registry.toml").write_text('schema_version = "1"\n', encoding="utf-8")
    (bench / "environment-profiles.toml").write_text(
        '[profiles.default]\nimage = "default"\n', encoding="utf-8"
    )
    suite_manifest = dataset_dir / "suite.toml"
    suite_manifest.write_text(
        'schema_version = "2"\n[dataset]\nid = "jacobian/alpha-v1"\n',
        encoding="utf-8",
    )
    member = members_dir / "alpha-task.toml"
    member.write_text('task_id = "alpha-task"\n', encoding="utf-8")
    task = SimpleNamespace(path=Path("alpha-task"))
    suite = SimpleNamespace(
        id="alpha-v1", path=dataset_dir, suite_manifest=suite_manifest, tasks=(task,)
    )
    return bench, suite
