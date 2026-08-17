"""Harbor verifier-support and committed-suite policy tests."""

from __future__ import annotations

from pathlib import Path

from benchmarks.tooling.harbor_suite import check_verifier_support

from tests.tooling.harbor_suite_support import (
    _make_suite_with_task,
)

ROOT = Path(__file__).resolve().parents[2]


def test_check_verifier_support_allows_task_owned_contents(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    (task / "tests" / "verifier_support.py").write_text("# task support\n")
    assert check_verifier_support(suite) == []


def test_check_verifier_support_reports_syntax_errors(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    (task / "tests" / "verifier_support.py").write_text("def broken(:\n")
    failures = check_verifier_support(suite)
    assert any("does not compile" in f for f in failures)


def test_check_verifier_support_rejects_support_symlink(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    support = task / "tests" / "verifier_support.py"
    support.unlink()
    support.symlink_to(task / "tests" / "verifier.py")
    failures = check_verifier_support(suite)
    assert any("regular, non-symlinked" in f for f in failures)


# ---------------------------------------------------------------------------
# Integration with committed datasets
# ---------------------------------------------------------------------------
