"""Architecture checker exclusions, orchestration, and diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.check_architecture import (
    ArchitecturePolicyError,
    assert_architecture,
    check_architecture,
)


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


_K = "knowledge"
_S = "search"
_RM = "Research" + "Memory"


def test_wt438_directory_is_excluded(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "wt-438/src/jacobian/bad.py",
        f"import subprocess\nfrom jacobian.memory import {_RM}\n"
        "import shutil\nshutil.which('git')\n"
        "import os\nenv = dict(os.environ)\n",
    )
    report = check_architecture(tmp_path)
    assert report.ok
    assert report.violations == ()


@pytest.mark.parametrize(
    "directory",
    [
        ".hypothesis",
        ".import_linter_cache",
        ".jacobian",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
    ],
)
def test_generated_directory_is_pruned(tmp_path: Path, directory: str) -> None:
    _write(
        tmp_path,
        f"{directory}/generated.py",
        "import subprocess\n",
    )
    _write(
        tmp_path,
        f"{directory}/generated.json",
        f'{{"operation_id": "{_K}.{_S}"}}\n',
    )

    report = check_architecture(tmp_path)

    assert report.ok, report.render()
    assert report.files_scanned == 0


def test_clean_tree_passes(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/bounded_process.py", "import subprocess\n")
    _write(
        tmp_path,
        "src/jacobian/domains/logic/operations.py",
        "from jacobian.bounded_process import run_bounded_process\n"
        "import shutil\nshutil.which('lean')\n",
    )
    _write(
        tmp_path,
        "src/jacobian/operation_dispatcher.py",
        "import os\nhome = os.environ.get('HOME')\n",
    )
    _write(
        tmp_path,
        "tests/boundary/process/test_bounded_process.py",
        "import subprocess\nsubprocess.run(['echo'])\n",
    )
    report = check_architecture(tmp_path)
    assert report.ok, report.render()


def test_report_render_shows_violations(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/bad.py", "import subprocess\n")
    report = check_architecture(tmp_path)
    rendered = report.render()
    assert "architecture:" in rendered
    assert "subprocess-confined" in rendered
    assert "src/jacobian/bad.py" in rendered


def test_assert_raises_on_failure(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/bad.py", "import subprocess\n")
    with pytest.raises(ArchitecturePolicyError, match="subprocess-confined"):
        assert_architecture(tmp_path)


def test_multiple_violations_sorted_by_path(tmp_path: Path) -> None:
    _write(tmp_path, "src/jacobian/zzz.py", "import subprocess\n")
    _write(tmp_path, "src/jacobian/aaa.py", "import subprocess\n")
    report = check_architecture(tmp_path)
    sub = [v for v in report.violations if v.code == "subprocess-confined"]
    assert sub[0].path < sub[1].path
