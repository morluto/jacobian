from __future__ import annotations

from pathlib import Path

import pytest
from tools.check_architecture import check_architecture


def _write(root: Path, relative: str, source: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _ownership_messages(root: Path) -> list[str]:
    return [
        violation.message
        for violation in check_architecture(root).violations
        if violation.code == "test-ownership"
    ]


@pytest.mark.parametrize(
    "historical_name", ["frontier", "migration", "regression", "release"]
)
def test_historical_composition_bucket_is_rejected(
    tmp_path: Path,
    historical_name: str,
) -> None:
    _write(tmp_path, f"tests/composition/runtime/test_{historical_name}.py")

    messages = _ownership_messages(tmp_path)

    assert any("historical status" in message for message in messages)


def test_unit_module_cannot_import_jacobian_runtime(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/unit/contracts/test_runtime.py",
        "from jacobian.runtime import CheckerAuthorityMode\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_domain_module_cannot_import_runtime_constructor(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_runtime.py",
        "from tests.support.catalog_build_runtime import create_catalog_build_runtime\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_domain_module_cannot_alias_runtime_import(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_runtime.py",
        "from tests.support.catalog_build_runtime import create_catalog_build_runtime as open_everything\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_domain_module_cannot_import_runtime_as_a_module(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_runtime.py",
        "import jacobian.runtime as runtime\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_unrelated_runtime_factory_call_is_accepted(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_factory.py",
        "def create_runtime():\n    return object()\n\ncreate_runtime()\n",
    )

    assert _ownership_messages(tmp_path) == []


def test_runtime_owned_passive_type_import_is_accepted(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_services.py",
        "from jacobian.runtime.resources import RuntimeResources\n",
    )

    assert _ownership_messages(tmp_path) == []


def test_domain_module_cannot_import_complete_runtime_fixtures(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_runtime.py",
        "from tests.support.complete_runtime_fixtures import fresh_complete_runtime\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_parent_fixture_cannot_import_jacobian_runtime(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/conftest.py",
        "from tests.support.catalog_build_runtime import create_catalog_build_runtime\n",
    )

    assert any(
        "complete-runtime fixtures" in message
        for message in _ownership_messages(tmp_path)
    )


def test_parent_fixture_cannot_reexport_complete_runtime_fixtures(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "tests/domain/conftest.py",
        "from tests.support.complete_runtime_fixtures import fresh_complete_runtime\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_composition_module_requires_a_semantic_owner_directory(tmp_path: Path) -> None:
    _write(tmp_path, "tests/composition/test_miscellaneous.py")

    assert _ownership_messages(tmp_path) == [
        "composition tests must declare portfolio, runtime, authority, "
        "interoperability, or CLI ownership in their directory"
    ]


def test_discarded_complete_runtime_fixture_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/boundary/providers/lean/startup/test_discard.py",
        "def test_hydration(authorized_complete_runtime, tmp_path):\n"
        "    _ = authorized_complete_runtime\n"
        "    create_catalog_build_runtime(tmp_path)\n",
    )

    assert _ownership_messages(tmp_path) == [
        "expensive runtime fixtures must be the subject under test, "
        "not discarded setup: authorized_complete_runtime"
    ]


@pytest.mark.parametrize(
    "source",
    [
        "from tests.support.catalog_build_runtime import create_catalog_build_runtime\n",
        "import tests.support.catalog_build_runtime as runtime\n",
        "from tests.support import catalog_build_runtime\n",
        "from tests.support.complete_runtime_fixtures import fresh_complete_runtime\n",
        "import tests.support.complete_runtime_fixtures as fixtures\n",
        "from tests.support import complete_runtime_fixtures\n",
    ],
)
def test_lean_behavior_module_cannot_import_complete_runtime(
    tmp_path: Path,
    source: str,
) -> None:
    _write(
        tmp_path,
        "tests/boundary/providers/lean/startup/test_lean.py",
        source,
    )

    assert _ownership_messages(tmp_path) == [
        "Lean behavior tests must use tests.support.lean_runtime "
        "instead of the complete catalog-build runtime"
    ]


@pytest.mark.parametrize(
    "source",
    [
        "import tests.support.catalog_build_runtime as runtime\n",
        "from tests.support import catalog_build_runtime\n",
        "import tests.support.complete_runtime_fixtures as fixtures\n",
        "from tests.support import complete_runtime_fixtures\n",
    ],
)
def test_domain_module_cannot_import_complete_runtime_as_a_module(
    tmp_path: Path,
    source: str,
) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_runtime.py",
        source,
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_cli_projection_tests_cannot_cold_start_the_complete_portfolio(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "tests/composition/cli/test_cli.py",
        "from jacobian.cli import app, create_cli_app\n"
        "from tests.support.catalog_build_runtime import create_catalog_build_runtime\n"
        "\n"
        "def test_cli_catalog_lists_operations(tmp_path):\n"
        "    create_catalog_build_runtime(tmp_path)\n"
        "    create_cli_app()\n"
        "    CliRunner().invoke(app, ['catalog'])\n",
    )

    messages = _ownership_messages(tmp_path)

    assert any(
        "CLI projection tests must use selected_runtime_opener" in message
        for message in messages
    )


def test_cli_init_complete_smoke_may_use_the_default_app(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/composition/cli/test_cli.py",
        "from jacobian.cli import app\n"
        "\n"
        "def test_cli_init_reports_installed_operation_count(tmp_path):\n"
        "    CliRunner().invoke(app, ['init'])\n",
    )

    assert _ownership_messages(tmp_path) == []


def test_owned_composition_module_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "tests/composition/interoperability/test_value_handoff.py")

    assert _ownership_messages(tmp_path) == []


def test_versioned_composition_module_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "tests/composition/runtime/test_mcp_sdk_2.py")

    assert _ownership_messages(tmp_path) == []
