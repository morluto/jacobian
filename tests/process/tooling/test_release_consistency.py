"""Tests for the release-surface consistency check (issue #1014)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from tests.process.tooling.ci import run_ci_script

ROOT = Path(__file__).resolve().parents[3]


def test_release_consistency_passes_on_clean_tree() -> None:
    """The checked-in tree must agree across all version-bearing surfaces."""

    result = run_ci_script("check-release-consistency", check=True)
    assert "All present release surfaces agree." in result.stdout


@pytest.mark.parametrize(
    "surface", ["npm/package-lock.json (top-level)", "pyproject.toml"]
)
def test_release_consistency_detects_version_drift(
    tmp_path: Path, surface: str
) -> None:
    """Version-bearing package surfaces cannot disagree with package.json."""
    npm_dir = tmp_path / "npm"
    npm_dir.mkdir()
    (npm_dir / "package.json").write_text(
        json.dumps({"name": "jacobian", "version": "0.11.0"}),
        encoding="utf-8",
    )
    lock_version = "0.10.0" if surface.startswith("npm/") else "0.11.0"
    (npm_dir / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "jacobian",
                "version": lock_version,
                "lockfileVersion": 3,
                "packages": {"": {"name": "jacobian", "version": lock_version}},
            }
        ),
        encoding="utf-8",
    )
    if surface == "pyproject.toml":
        (tmp_path / "pyproject.toml").write_text(
            textwrap.dedent(
                """
                [project]
                name = "jacobian"
                version = "0.10.0"
                """
            ).strip(),
            encoding="utf-8",
        )

    result = run_ci_script("check-release-consistency", "--root", tmp_path, check=False)
    assert result.returncode == 1
    assert surface in result.stderr
    if surface.startswith("npm/"):
        assert "0.10.0 (expected 0.11.0)" in result.stderr


def test_release_consistency_accepts_explicit_expected(tmp_path: Path) -> None:
    """--expected overrides the npm/package.json derivation."""

    npm_dir = tmp_path / "npm"
    npm_dir.mkdir()
    (npm_dir / "package.json").write_text(
        json.dumps({"name": "jacobian", "version": "0.10.0"}), encoding="utf-8"
    )
    (npm_dir / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "jacobian",
                "version": "0.10.0",
                "lockfileVersion": 3,
                "packages": {"": {"name": "jacobian", "version": "0.10.0"}},
            }
        ),
        encoding="utf-8",
    )

    result = run_ci_script(
        "check-release-consistency",
        "--root",
        tmp_path,
        "--expected",
        "0.10.0",
        check=True,
    )
    assert "All present release surfaces agree." in result.stdout
