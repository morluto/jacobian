"""Harbor verifier-support and committed-suite policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.tooling.harbor_suite import check_verifier_support
from tests.unit.tooling.harbor_suite_support import (
    _make_suite_with_task,
    patch_harbor_root,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def patched_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    return patch_harbor_root(monkeypatch, tmp_path)


def test_check_verifier_support_passes_when_all_copies_match(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    support = "# canonical support\n"
    (patched_root / "benchmarks" / "tooling" / "verifier_support.py").write_text(
        support
    )
    (task / "tests" / "verifier_support.py").write_text(support)
    assert check_verifier_support(suite) == []


def test_check_verifier_support_reports_drift(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    (patched_root / "benchmarks" / "tooling" / "verifier_support.py").write_text(
        "# canonical\n"
    )
    (task / "tests" / "verifier_support.py").write_text("# drifted\n")
    failures = check_verifier_support(suite)
    assert any("differs" in f for f in failures)


def test_check_verifier_support_uses_repository_canonical_copy(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, _task = _make_suite_with_task(tmp_path)
    # No dataset-root verifier_support.py is needed: the repository-owned
    # canonical copy is the only source of truth.
    assert check_verifier_support(suite) == []


# ---------------------------------------------------------------------------
# Integration with committed datasets
# ---------------------------------------------------------------------------
