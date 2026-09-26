"""Tests for the explicitly scoped Harbor verifier checksum updater."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.tooling.errors import HarborSuiteError
from tools import sync_harbor_verifier_support as checksum_tool

from tests.tooling.harbor_suite_support import (
    _make_canonical_task,
    _make_suite_with_task,
    patch_harbor_root,
)


def test_checksum_update_only_rewrites_selected_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = patch_harbor_root(monkeypatch, tmp_path)
    suite, first = _make_suite_with_task(tmp_path)
    second = _make_canonical_task(tmp_path, task_id="test-v1-b")
    first_ref = suite.tasks[0]
    second_ref = replace(
        first_ref,
        name="jacobian/test-v1-b",
        path=second,
    )
    scoped_suite = replace(suite, tasks=(first_ref, second_ref))
    monkeypatch.setattr(checksum_tool, "ROOT", root)
    monkeypatch.setattr(checksum_tool, "get_suite", lambda _dataset: scoped_suite)

    first_docker = first / "tests" / "Dockerfile"
    second_docker = second / "tests" / "Dockerfile"
    second_before = second_docker.read_bytes()
    checksum_tool.update("test-v1", ("test-v1-a",))

    assert (
        'jacobian.checksum="2a8f28454f793b040a9c25967b59fe5c65c3149889d95010a2a95faddea27a47"'
        in first_docker.read_text()
    )
    assert second_docker.read_bytes() == second_before


def test_checksum_update_requires_a_nonempty_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_harbor_root(monkeypatch, tmp_path)
    suite, _task = _make_suite_with_task(tmp_path)
    monkeypatch.setattr(checksum_tool, "get_suite", lambda _dataset: suite)

    with pytest.raises(HarborSuiteError, match="at least one task"):
        checksum_tool.update("test-v1", ())


def test_support_sync_is_explicit_and_limited_to_selected_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = patch_harbor_root(monkeypatch, tmp_path)
    suite, first = _make_suite_with_task(tmp_path)
    second = _make_canonical_task(tmp_path, task_id="test-v1-b")
    first_ref = suite.tasks[0]
    second_ref = replace(first_ref, name="jacobian/test-v1-b", path=second)
    scoped_suite = replace(suite, tasks=(first_ref, second_ref))
    template = tmp_path / "template.py"
    template.write_text("# current support\n", encoding="utf-8")
    second_support = second / "tests" / "verifier_support.py"
    second_before = second_support.read_bytes()

    monkeypatch.setattr(checksum_tool, "ROOT", root)
    monkeypatch.setattr(checksum_tool, "SUPPORT_TEMPLATE", template)
    monkeypatch.setattr(checksum_tool, "get_suite", lambda _dataset: scoped_suite)

    checksum_tool.update("test-v1", ("test-v1-a",), write_support=True)

    assert (
        first / "tests" / "verifier_support.py"
    ).read_text() == "# current support\n"
    assert second_support.read_bytes() == second_before
