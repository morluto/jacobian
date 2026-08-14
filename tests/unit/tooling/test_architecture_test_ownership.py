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

    assert any(
        "historical status" in message for message in _ownership_messages(tmp_path)
    )


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
