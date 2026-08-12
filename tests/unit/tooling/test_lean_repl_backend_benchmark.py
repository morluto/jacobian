from __future__ import annotations

import pytest
from benchmarks.tooling import lean_repl_backend_benchmark as benchmark_module


def test_lean_repl_benchmark_refuses_dirty_tracked_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_module,
        "git_tracked_worktree_is_clean",
        lambda _root: False,
    )

    with pytest.raises(SystemExit, match="clean tracked worktree"):
        benchmark_module._source_sha()


def test_lean_repl_benchmark_requires_a_source_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_module,
        "git_tracked_worktree_is_clean",
        lambda _root: True,
    )
    monkeypatch.setattr(benchmark_module, "git_head_sha", lambda _root: None)

    with pytest.raises(SystemExit, match="source revision"):
        benchmark_module._source_sha()


def test_lean_repl_benchmark_binds_clean_source_revision(
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
