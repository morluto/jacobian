from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[3]


def _recipe(makefile: str, target: str, following: str) -> str:
    return makefile.split(f"{target}:", 1)[1].split(f"{following}:", 1)[0]


def test_lean_lane_is_serial_and_has_its_own_deadline() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    lean = _recipe(makefile, "test-lean", "test-e2e")

    assert "pytest_lifecycle.py" in lean or "PYTEST_RUNNER" in lean
    assert "--timeout=300" in lean
    assert "--timeout-method=signal" in lean
    assert "-n " not in lean


def test_process_lane_uses_bounded_parallelism() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    process = _recipe(makefile, "test-process", "test-mcp")

    assert "PYTEST_RUNNER" in process or "pytest_lifecycle.py" in process
    assert "-n 2" in process
    assert "--timeout=120" in process
    assert "--timeout-method=signal" in process


def test_mcp_lane_is_supervised_like_process() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    mcp = _recipe(makefile, "test-mcp", "test-provider")

    assert "PYTEST_RUNNER" in mcp or "pytest_lifecycle.py" in mcp
    assert "-n 2" in mcp
    assert "--timeout-method=signal" in mcp


def test_ordinary_check_dispatches_to_fixed_direct_pytest_lanes() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    ordinary = _recipe(makefile, "test-ordinary", "test-compatibility")

    assert "ORDINARY_TEST_LANES" in ordinary
    assert "PYTEST_RUNNER" not in ordinary


def test_fixed_ordinary_lanes_cover_the_default_testpaths() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    paths = config["tool"]["pytest"]["ini_options"]["testpaths"]

    assert paths == [
        "tests/unit",
        "tests/component",
        "tests/domain",
        "tests/composition",
        "tests/boundary/providers/cvc5",
        "tests/boundary/providers/external_sat",
        "tests/boundary/providers/flint",
        "tests/e2e",
    ]
    assert "tests/unit" in _recipe(makefile, "test-unit", "test-component")
    assert "tests/component" in _recipe(makefile, "test-component", "test-domain")
    assert "tests/domain" in _recipe(makefile, "test-domain", "test-composition")
    assert "tests/composition" in _recipe(makefile, "test-composition", "test-storage")
    provider = _recipe(makefile, "test-provider", "test-lean")
    assert all(path in provider for path in paths[4:7])
    assert "tests/e2e" in _recipe(makefile, "test-e2e", "test-ordinary")


def test_default_testpaths_omit_lean_and_supervised_boundaries() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    paths = config["tool"]["pytest"]["ini_options"]["testpaths"]

    assert "tests/unit" in paths
    assert "tests/domain" in paths
    assert "tests/boundary/providers/cvc5" in paths
    assert "tests/boundary/providers/lean" not in paths
    assert "tests/boundary/storage" not in paths
    assert "tests/boundary/process" not in paths
    assert "tests/boundary/mcp" not in paths
    assert all("lean" not in path for path in paths)
