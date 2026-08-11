"""Tests for the explicitly scoped Harbor verifier checksum updater."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import verifier_bundle_checksum
from tests.unit.tooling.harbor_suite_support import (
    _make_canonical_task,
    _make_suite_with_task,
    patch_harbor_root,
)
from tools import sync_harbor_verifier_support as checksum_tool


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
    verifier_digest = verifier_bundle_checksum(first / "tests")

    checksum_tool.update("test-v1", ("test-v1-a",))

    assert f'jacobian.checksum="{verifier_digest}"' in first_docker.read_text()
    assert second_docker.read_bytes() == second_before


def test_checksum_update_requires_a_nonempty_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_harbor_root(monkeypatch, tmp_path)
    suite, _task = _make_suite_with_task(tmp_path)
    monkeypatch.setattr(checksum_tool, "get_suite", lambda _dataset: suite)

    with pytest.raises(HarborSuiteError, match="at least one task"):
        checksum_tool.update("test-v1", ())
