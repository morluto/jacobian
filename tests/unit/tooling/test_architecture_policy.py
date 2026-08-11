from __future__ import annotations

import json
from pathlib import Path

from tools.check_test_architecture import (
    ArchitecturePolicyError,
    check_test_architecture,
    load_topology_manifest,
)


def _test_file(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_lower_tier_rules_report_actionable_locations(tmp_path: Path) -> None:
    _test_file(
        tmp_path,
        "tests/unit/test_bad.py",
        """
import sqlite3
import subprocess
import sympy
from jacobian.runtime import create_runtime
from jacobian.portfolio import BUILTIN_PORTFOLIO
from jacobian.storage.repository import ArtifactRepository
""",
    )

    report = check_test_architecture(tmp_path)

    assert not report.ok
    assert {(item.code, item.path) for item in report.violations} == {
        ("sqlite-unit", "tests/unit/test_bad.py"),
        ("process-unit", "tests/unit/test_bad.py"),
        ("provider-import", "tests/unit/test_bad.py"),
        ("runtime-usage", "tests/unit/test_bad.py"),
        ("builtin-portfolio", "tests/unit/test_bad.py"),
    }
    assert all(item.line is not None for item in report.violations)
    assert "tests/unit/test_bad.py:2" in report.render()


def test_runtime_is_allowed_only_at_explicit_lifecycle_boundaries(
    tmp_path: Path,
) -> None:
    _test_file(
        tmp_path,
        "tests/composition/test_runtime.py",
        "from jacobian.runtime import create_runtime\n\ndef test_runtime():\n    create_runtime(None)\n",
    )
    _test_file(
        tmp_path,
        "tests/boundary/runtime/startup/test_runtime.py",
        "from jacobian.runtime import create_runtime\n",
    )
    _test_file(
        tmp_path,
        "tests/e2e/test_workflow.py",
        "from jacobian.runtime import create_runtime\n",
    )
    _test_file(
        tmp_path,
        "tests/domain/test_domain.py",
        "from jacobian.runtime import create_runtime\n",
    )

    report = check_test_architecture(tmp_path)

    assert [(item.code, item.path) for item in report.violations] == [
        ("runtime-usage", "tests/domain/test_domain.py")
    ]


def test_provider_imports_allow_boundary_and_focused_component(tmp_path: Path) -> None:
    _test_file(tmp_path, "tests/boundary/providers/test_z3.py", "import z3\n")
    _test_file(tmp_path, "tests/component/providers/test_sympy.py", "import sympy\n")
    _test_file(tmp_path, "tests/domain/test_z3.py", "import z3\n")
    _test_file(tmp_path, "tests/component/test_accidental.py", "import cvc5\n")

    report = check_test_architecture(tmp_path)

    assert [(item.code, item.path) for item in report.violations] == [
        ("provider-import", "tests/component/test_accidental.py"),
        ("provider-import", "tests/domain/test_z3.py"),
    ]


def test_root_conftest_rejects_high_cost_fixture(tmp_path: Path) -> None:
    _test_file(
        tmp_path,
        "tests/conftest.py",
        "import pytest\n\n@pytest.fixture\ndef fresh_complete_runtime():\n    yield object()\n",
    )

    report = check_test_architecture(tmp_path)

    assert [item.code for item in report.violations] == ["root-high-cost-fixture"]


def test_sibling_conftest_imports_are_rejected(tmp_path: Path) -> None:
    _test_file(
        tmp_path,
        "tests/boundary/conftest.py",
        "from tests.composition.conftest import complete_runtime\n",
    )

    report = check_test_architecture(tmp_path)

    assert [item.code for item in report.violations] == ["conftest-import"]


def test_runtime_fixture_builders_are_explicit_construction_owners(
    tmp_path: Path,
) -> None:
    for name in ("runtime_templates.py", "runtime_profiles.py"):
        _test_file(
            tmp_path,
            f"tests/support/{name}",
            "from jacobian.runtime import create_runtime\n",
        )

    assert check_test_architecture(tmp_path).ok


def test_support_modules_cannot_hide_complete_runtime_construction(
    tmp_path: Path,
) -> None:
    """Generic support code must not become an unowned composition seam."""

    _test_file(
        tmp_path,
        "tests/support/runtime_helper.py",
        "from jacobian.runtime import create_runtime\n\n"
        "def open_runtime(root):\n    return create_runtime(root)\n",
    )

    report = check_test_architecture(tmp_path)

    assert [item.code for item in report.violations] == ["runtime-usage"]
    assert report.violations[0].path == "tests/support/runtime_helper.py"


def test_topology_manifest_requires_exactly_one_lane(tmp_path: Path) -> None:
    _test_file(tmp_path, "tests/unit/test_owned.py", "def test_owned(): pass\n")
    _test_file(tmp_path, "tests/misc/test_unowned.py", "def test_unowned(): pass\n")
    (tmp_path / "tests" / "topology.toml").write_text(
        """
[[lanes]]
name = "unit"
owned_paths = ["tests/unit/**"]
[[lanes]]
name = "component"
paths = ["tests/unit/**", "tests/component/**"]
""",
        encoding="utf-8",
    )

    report = check_test_architecture(tmp_path)

    assert [(item.code, item.path) for item in report.violations] == [
        ("lane-ownership", "tests/misc/test_unowned.py"),
        ("lane-ownership", "tests/unit/test_owned.py"),
    ]
    manifest = load_topology_manifest(tmp_path / "tests/topology.toml")
    assert manifest is not None
    assert manifest.owners("tests/unit/test_owned.py") == ("unit", "component")


def test_ratchet_mode_fails_only_for_new_violations(tmp_path: Path) -> None:
    _test_file(tmp_path, "tests/unit/test_bad.py", "import sqlite3\n")
    strict = check_test_architecture(tmp_path)
    assert strict.failed
    baseline = {
        "violations": [
            {
                "path": "tests/unit/test_bad.py",
                "line": 1,
                "code": "sqlite-unit",
            }
        ]
    }
    report = check_test_architecture(tmp_path, mode="ratchet", baseline=baseline)
    assert report.violations
    assert report.new_violations == ()
    assert report.ok

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert check_test_architecture(tmp_path, mode="ratchet", baseline=baseline_path).ok


def test_report_error_preserves_rendered_diagnostics(tmp_path: Path) -> None:
    _test_file(tmp_path, "tests/unit/test_bad.py", "import sqlite3\n")
    report = check_test_architecture(tmp_path)
    assert report.failed
    try:
        from tools.check_test_architecture import assert_test_architecture

        assert_test_architecture(tmp_path)
    except ArchitecturePolicyError as exc:
        assert str(exc) == report.render()
    else:  # pragma: no cover - assertion is the test's purpose
        raise AssertionError("expected architecture policy failure")
