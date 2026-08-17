"""Shared fixtures for Harbor tooling tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tooling.harbor_suite_support import patch_harbor_root


@pytest.fixture
def patched_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    return patch_harbor_root(monkeypatch, tmp_path)
