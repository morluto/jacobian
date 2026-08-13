from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderDigestKind,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.contracts.provider_measurements import (
    ProviderInstalledSize,
    ProviderMeasurementStatus,
)
from jacobian.provider_measurements import (
    _PYTHON_PROBE,
    _child_peak_rss_bytes,
    _installed_size,
    _measure_command,
    _process_environment,
    measure_provider,
)


def _runtime_with_missing_distribution() -> ProviderObservation:
    return ProviderObservation(
        provider="tests.fixture",
        availability=ProviderAvailability.AVAILABLE,
        version="1.2.3",
        digest="sha256:" + "a" * 64,
        digest_kind=ProviderDigestKind.PYTHON_DISTRIBUTION_RECORD,
        platform="linux-x86_64",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        configuration={"distribution": "missing-provider-distribution"},
    )


def test_child_peak_rss_prefers_marker_over_sampled_value() -> None:
    stdout = (
        b"JACOBIAN_MEASUREMENT_RSS_BYTES=1\n"
        b"noise\nJACOBIAN_MEASUREMENT_RSS_BYTES=12345\n"
    )

    assert _child_peak_rss_bytes(stdout, sampled=0) == 12345
    assert _child_peak_rss_bytes(stdout, sampled=99999) == 12345


def test_child_peak_rss_falls_back_to_sampled_without_marker() -> None:
    stdout = b"no marker here\n"

    assert _child_peak_rss_bytes(stdout, sampled=4096) == 4096
    assert _child_peak_rss_bytes(stdout, sampled=None) is None


def test_child_peak_rss_ignores_malformed_marker() -> None:
    stdout = b"JACOBIAN_MEASUREMENT_RSS_BYTES=\n"

    assert _child_peak_rss_bytes(stdout, sampled=2048) == 2048


def test_python_probe_emits_rss_marker_for_short_completed_child() -> None:
    # A short probe that exits well before the engine's procfs sampler can poll
    # it still reports a trustworthy positive peak RSS, sourced from the
    # child's own RUSAGE_SELF rather than cumulative prior-child rusage.
    sample = _measure_command(
        [sys.executable, "-c", _PYTHON_PROBE, "jacobian.canonical", "cold-start"]
    )

    assert sample.status is ProviderMeasurementStatus.COMPLETED
    assert sample.peak_rss_bytes is not None
    assert sample.peak_rss_bytes > 0


def test_python_probe_fails_closed_under_optimization(tmp_path) -> None:
    (tmp_path / "networkx.py").write_text(
        "def path_graph(_count):\n"
        "    return object()\n\n"
        "def is_connected(_graph):\n"
        "    return False\n"
    )
    environment = _process_environment()
    environment["PYTHONPATH"] = str(tmp_path)

    sample = _measure_command(
        [
            sys.executable,
            "-O",
            "-c",
            _PYTHON_PROBE,
            "jacobian.networkx",
            "reproduction",
        ],
        environment=environment,
    )

    assert sample.status is ProviderMeasurementStatus.ERROR


def test_measure_command_without_marker_falls_back_to_engine_sample() -> None:
    sample = _measure_command([sys.executable, "-c", "pass"])

    assert sample.status is ProviderMeasurementStatus.COMPLETED
    # No marker is emitted, so the value is whatever the engine sampled; the
    # contract is that the fallback path does not raise and stays non-negative.
    assert sample.peak_rss_bytes is None or sample.peak_rss_bytes >= 0


def test_installed_size_reports_missing_distribution_metadata() -> None:
    measurement = _installed_size(_runtime_with_missing_distribution())

    assert measurement.status is ProviderMeasurementStatus.ERROR
    assert measurement.bytes is None
    assert measurement.detail == "The provider distribution metadata is unavailable."


def test_installed_size_rejects_missing_distribution_file_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.provider_measurements.importlib.metadata.distribution",
        lambda _name: SimpleNamespace(files=None),
    )

    measurement = _installed_size(_runtime_with_missing_distribution())

    assert measurement.status is ProviderMeasurementStatus.ERROR
    assert measurement.bytes is None
    assert (
        measurement.detail == "The provider distribution file manifest is unavailable."
    )


def test_installed_size_rejects_unreadable_distribution_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    file_path = tmp_path / "provider.py"
    file_path.write_text("value = 1\n")
    monkeypatch.setattr(
        "jacobian.provider_measurements.importlib.metadata.distribution",
        lambda _name: SimpleNamespace(
            files=("provider.py",),
            locate_file=lambda _path: file_path,
        ),
    )
    monkeypatch.setattr(type(file_path), "is_file", lambda _path: True)
    monkeypatch.setattr(type(file_path), "is_symlink", lambda _path: False)
    monkeypatch.setattr(
        type(file_path),
        "stat",
        lambda _path: (_ for _ in ()).throw(OSError("unreadable")),
    )

    measurement = _installed_size(_runtime_with_missing_distribution())

    assert measurement.status is ProviderMeasurementStatus.ERROR
    assert measurement.bytes is None
    assert measurement.detail == "The provider installed-size measurement failed."


def test_provider_measurement_reports_missing_distribution_metadata() -> None:
    measurement = measure_provider(_runtime_with_missing_distribution())

    assert measurement.installed_size.status is ProviderMeasurementStatus.ERROR
    assert measurement.installed_size.bytes is None
    assert (
        measurement.installed_size.detail
        == "The provider distribution metadata is unavailable."
    )


def test_installed_size_contract_requires_a_value_or_diagnostic() -> None:
    with pytest.raises(
        ValueError,
        match="completed installed-size measurement requires bytes",
    ):
        ProviderInstalledSize(status=ProviderMeasurementStatus.COMPLETED)

    with pytest.raises(
        ValueError,
        match="incomplete installed-size measurement requires a detail",
    ):
        ProviderInstalledSize(status=ProviderMeasurementStatus.ERROR)

    with pytest.raises(
        ValueError,
        match="incomplete installed-size measurement must not include bytes",
    ):
        ProviderInstalledSize(
            status=ProviderMeasurementStatus.ERROR,
            bytes=1,
            detail="The measurement failed.",
        )
