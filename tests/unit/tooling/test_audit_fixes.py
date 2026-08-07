"""Formal tests for the audit-confirmed bugs and their fixes.

These tests serve as formal validation that:
1. _observation_pair_failures fails closed on malformed JSON (Bug 1a)
2. _usage rejects non-dict stats (Bug 1b)
3. _load_registry_cache provides invalidation (Bug 3a)
4. install_source_only_importer purges sys.modules (Bug TOCTOU)
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest


# Bug 1a: _observation_pair_failures should fail closed on non-dict JSON
def test_observation_pair_failures_fails_closed_on_non_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Formally prove that malformed observation JSON is not silently accepted."""
    from benchmarks.tooling import benchmark_contracts

    # Test: JSON array instead of object
    def mock_read_json_array(path):
        return []

    monkeypatch.setattr(benchmark_contracts, "_read_json", mock_read_json_array)
    failures = benchmark_contracts._observation_pair_failures()
    assert len(failures) > 0, "Bug 1a: malformed JSON should not silently pass"
    assert "malformed" in failures[0].lower()

    # Test: JSON null
    def mock_read_json_null(path):
        return None

    monkeypatch.setattr(benchmark_contracts, "_read_json", mock_read_json_null)
    failures = benchmark_contracts._observation_pair_failures()
    assert len(failures) > 0, "Bug 1a: null JSON should not silently pass"


# Bug 1b: _usage should reject non-dict stats
def test_usage_rejects_non_dict_stats() -> None:
    """Formally prove that non-dict stats is rejected."""
    from benchmarks.tooling import heldout_runner

    # Create a temporary result file with stats as null
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"stats": None}, f)
        path = Path(f.name)

    try:
        with pytest.raises(Exception) as exc_info:
            heldout_runner._usage(path)
        assert "stats" in str(exc_info.value).lower()
    finally:
        path.unlink(missing_ok=True)


# Bug 3a: Registry cache should be invalidatable
def test_registry_cache_invalidation() -> None:
    """Formally prove that cache invalidation works."""
    from benchmarks.tooling import harbor_suite

    assert hasattr(harbor_suite, "invalidate_registry_cache"), (
        "invalidate_registry_cache should exist"
    )
    harbor_suite.invalidate_registry_cache()
    assert not harbor_suite._load_registry_cache, (
        "Cache should be empty after invalidation"
    )


# Bug TOCTOU: install_source_only_importer should purge sys.modules
def test_source_only_importer_purges_sys_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Formally prove that pre-imported modules are purged."""
    from jacobian.implementation import _SourceOnlyFinder, install_source_only_importer

    # Create a fake module in sys.modules that looks like the target package
    fake_module = type(sys)("fake_test_package")
    fake_module.some_value = "stale"
    monkeypatch.setitem(sys.modules, "fake_test_package", fake_module)
    monkeypatch.setitem(
        sys.modules, "fake_test_package.helper", type(sys)("fake_test_package.helper")
    )

    install_source_only_importer("fake_test_package:main")
    assert "fake_test_package" not in sys.modules, (
        "Pre-imported module should be purged"
    )
    assert "fake_test_package.helper" not in sys.modules, (
        "Pre-imported submodule should be purged"
    )

    # Clean up meta_path — monkeypatch does not manage this automatically
    # since the importer appends to sys.meta_path after import.
    monkeypatch.setattr(
        sys, "meta_path",
        [f for f in sys.meta_path if not isinstance(f, _SourceOnlyFinder)],
    )
