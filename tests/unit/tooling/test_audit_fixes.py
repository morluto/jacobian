"""Regression tests for malformed benchmark data and source-only imports."""

import json
import sys
from pathlib import Path

import pytest


def test_observation_pair_failures_fails_closed_on_non_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.tooling import benchmark_contracts

    def mock_read_json_array(path):
        return []

    monkeypatch.setattr(benchmark_contracts, "_read_json", mock_read_json_array)
    failures = benchmark_contracts._observation_pair_failures()
    assert any("malformed" in failure.lower() for failure in failures)

    def mock_read_json_null(path):
        return None

    monkeypatch.setattr(benchmark_contracts, "_read_json", mock_read_json_null)
    failures = benchmark_contracts._observation_pair_failures()
    assert any("malformed" in failure.lower() for failure in failures)


def test_usage_rejects_non_dict_stats(tmp_path: Path) -> None:
    from benchmarks.tooling import heldout_runner
    from benchmarks.tooling.errors import HarborSuiteError

    path = tmp_path / "result.json"
    path.write_text(json.dumps({"stats": None}), encoding="utf-8")

    with pytest.raises(HarborSuiteError, match="stats must be an object"):
        heldout_runner._usage(path)


def test_source_only_importer_purges_sys_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.implementation import install_source_only_importer

    # Restore the finder-free list after the test.
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))

    fake_module = type(sys)("fake_test_package")
    fake_module.some_value = "stale"
    monkeypatch.setitem(sys.modules, "fake_test_package", fake_module)
    monkeypatch.setitem(
        sys.modules, "fake_test_package.helper", type(sys)("fake_test_package.helper")
    )

    install_source_only_importer("fake_test_package:main")
    assert "fake_test_package" not in sys.modules
    assert "fake_test_package.helper" not in sys.modules
