"""Behavioral tests for complete-runtime inventory reporting."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.inventory_test_runtime import inventory_modules, main


def _write_topology(root: Path) -> None:
    tests = root / "tests"
    tests.mkdir()
    (tests / "topology.toml").write_text(
        """
[[lanes]]
name = "composition"
paths = ["tests/composition"]
""".lstrip(),
        encoding="utf-8",
    )


def _write_test(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_inventory_reports_owner_resources_fixtures_and_setup_weight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_topology(tmp_path)
    _write_test(
        tmp_path,
        "tests/composition/test_runtime.py",
        """
import sqlite3

def test_runtime(attached_complete_runtime):
    assert sqlite3 is not None
""".lstrip(),
    )

    (row,) = inventory_modules(tmp_path)

    assert row.path == "tests/composition/test_runtime.py"
    assert row.semantic_owner == "composition"
    assert row.complete_runtime_fixtures == ("attached_complete_runtime",)
    assert row.resource_imports == ("sqlite3",)
    assert row.setup_weight == 40
    assert not row.unjustified_authorized

    assert main([str(tmp_path)]) == 0
    summary = capsys.readouterr().out
    assert "with_complete_runtime=1" in summary
    assert "with_resource_imports=1" in summary
    assert "setup_weight_total=40" in summary


def test_authorized_runtime_with_verification_behavior_is_justified(
    tmp_path: Path,
) -> None:
    _write_topology(tmp_path)
    _write_test(
        tmp_path,
        "tests/composition/test_verified.py",
        """
def test_verified(authorized_complete_runtime):
    result = authorized_complete_runtime.services.verification.verify_witness()
    assert result.verification_record is not None
""".lstrip(),
    )

    (row,) = inventory_modules(tmp_path)

    assert row.has_verify_signal
    assert not row.unjustified_authorized


def test_authorized_runtime_without_verification_behavior_is_flagged(
    tmp_path: Path,
) -> None:
    _write_topology(tmp_path)
    _write_test(
        tmp_path,
        "tests/composition/test_catalog.py",
        """
def test_catalog(authorized_complete_runtime):
    assert authorized_complete_runtime.portfolio is not None
""".lstrip(),
    )

    (row,) = inventory_modules(tmp_path)

    assert not row.has_verify_signal
    assert row.unjustified_authorized
