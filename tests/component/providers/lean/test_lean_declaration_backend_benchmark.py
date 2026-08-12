from __future__ import annotations

import pytest
from benchmarks.tooling import lean_declaration_backend_benchmark as benchmark_module

from jacobian.contracts.lean import LeanEnvironment
from jacobian.lean_frontend.declaration_protocol import (
    LeanDeclarationInspectQuery,
    LeanDeclarationSearchQuery,
)


def test_lean_declaration_benchmark_refuses_dirty_tracked_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_module,
        "git_tracked_worktree_is_clean",
        lambda _root: False,
    )

    with pytest.raises(SystemExit, match="clean tracked worktree"):
        benchmark_module._source_sha()


def test_lean_declaration_benchmark_binds_clean_source_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    monkeypatch.setattr(
        benchmark_module,
        "git_tracked_worktree_is_clean",
        lambda _root: True,
    )
    monkeypatch.setattr(benchmark_module, "git_head_sha", lambda _root: revision)

    assert benchmark_module._source_sha() == revision


def test_lean_declaration_benchmark_uses_narrow_exact_workloads() -> None:
    core = benchmark_module._query(LeanEnvironment.CORE, "inspect")
    mathlib = benchmark_module._query(LeanEnvironment.MATHLIB, "search")

    assert isinstance(core, LeanDeclarationInspectQuery)
    assert core.declaration_name == "Nat.add"
    assert core.target_module_prefixes == ("Init",)
    assert isinstance(mathlib, LeanDeclarationSearchQuery)
    assert mathlib.name_contains == "irrational_sqrt_two"
    assert mathlib.limit == 1


def test_lean_declaration_benchmark_validates_cell_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = {
        "JACOBIAN_LEAN_DECL_BENCH_ENVIRONMENT": "MATHLIB",
        "JACOBIAN_LEAN_DECL_BENCH_BACKEND": "persistent",
        "JACOBIAN_LEAN_DECL_BENCH_OPERATION": "inspect",
    }
    monkeypatch.setattr(benchmark_module.os, "environ", settings)

    assert benchmark_module._settings() == (
        LeanEnvironment.MATHLIB,
        "persistent",
        "inspect",
    )
