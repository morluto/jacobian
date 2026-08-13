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


def test_unit_module_cannot_create_a_runtime(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/unit/contracts/test_runtime.py",
        "from jacobian.runtime import create_runtime\ncreate_runtime('state')\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_domain_module_cannot_create_a_complete_runtime(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_runtime.py",
        "from jacobian.runtime import create_runtime\ncreate_runtime('state')\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_domain_module_cannot_alias_complete_runtime_construction(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_runtime.py",
        "from jacobian.runtime import create_runtime as open_everything\n"
        "open_everything('state')\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_domain_module_cannot_import_runtime_as_a_module(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_runtime.py",
        "import jacobian.runtime as runtime\nruntime.create_runtime('state')\n",
    )

    assert _ownership_messages(tmp_path) == [
        "focused tests must use their owning seam instead of the complete runtime"
    ]


def test_unrelated_create_runtime_call_is_accepted(tmp_path: Path) -> None:
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
        "from jacobian.runtime.services import CoreServices\n",
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


def test_parent_fixture_cannot_construct_complete_runtime(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/conftest.py",
        "from jacobian.runtime import create_runtime\ncreate_runtime('state')\n",
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


def test_owned_composition_module_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "tests/composition/interoperability/test_value_handoff.py")

    assert _ownership_messages(tmp_path) == []


def test_versioned_composition_module_is_accepted(tmp_path: Path) -> None:
    _write(tmp_path, "tests/composition/runtime/test_mcp_sdk_2.py")

    assert _ownership_messages(tmp_path) == []
