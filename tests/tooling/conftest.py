"""Shared fixtures for Harbor tooling tests.

Three independent fixture resources are provided:

* ``synthetic_harbor_root`` — filesystem root/path containment only.
* ``synthetic_profiles`` — immutable profile mapping value only.
* ``git_initialized_root`` — only for tests that exercise real Git ignore behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tooling.harbor_suite_support import (
    patch_harbor_root,
    patch_harbor_root_with_git,
)


@pytest.fixture
def synthetic_harbor_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect harbor_suite.ROOT to tmp_path (no Git, no profiles)."""
    return patch_harbor_root(monkeypatch, tmp_path)


@pytest.fixture
def git_initialized_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect harbor_suite.ROOT to tmp_path with a real Git repository."""
    return patch_harbor_root_with_git(monkeypatch, tmp_path)
