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
    "historical_name", ["frontier", "migration", "regression", "release", "1373"]
)
def test_historical_composition_bucket_is_rejected(
    tmp_path: Path,
    historical_name: str,
) -> None:
    _write(tmp_path, f"tests/composition/runtime/test_{historical_name}.py")

    messages = _ownership_messages(tmp_path)

    assert any("historical status" in message for message in messages)


def test_domain_module_cannot_install_multiple_domain_bundles(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/domain/polynomial/test_mixed.py",
        "from jacobian.domains.polynomial import build_polynomial_bundle\n"
        "from jacobian.domains.number_theory import build_number_theory_bundle\n",
    )

    assert _ownership_messages(tmp_path) == [
        "domain tests may install bundles from only one domain owner"
    ]


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
