from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError

import pytest

import jacobian.providers.metadata as provider_metadata


@dataclass
class _Distribution:
    requested_name: str

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": f"recorded-{self.requested_name}"}

    @property
    def version(self) -> str:
        return f"1.0-{self.requested_name}"


@pytest.fixture
def isolated_metadata_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_metadata, "_version_cache", {})
    monkeypatch.setattr(provider_metadata, "_summary_cache", {})


def test_distribution_summary_computes_identical_identity_once(
    isolated_metadata_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_metadata_cache
    requested: list[str] = []

    def lookup(name: str) -> _Distribution:
        requested.append(name)
        return _Distribution(name)

    monkeypatch.setattr(provider_metadata, "distribution", lookup)

    first = provider_metadata.distribution_summary("provider-alpha")
    second = provider_metadata.distribution_summary("provider-alpha")

    assert second is first
    assert requested == ["provider-alpha"]


def test_distribution_summary_keeps_distinct_inputs_distinct(
    isolated_metadata_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_metadata_cache
    requested: list[str] = []

    def lookup(name: str) -> _Distribution:
        requested.append(name)
        return _Distribution(name)

    monkeypatch.setattr(provider_metadata, "distribution", lookup)

    alpha = provider_metadata.distribution_summary("provider-alpha")
    beta = provider_metadata.distribution_summary("provider-beta")

    assert alpha != beta
    assert requested == ["provider-alpha", "provider-beta"]


def test_missing_distribution_is_not_cached(
    isolated_metadata_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_metadata_cache
    calls = 0

    def lookup(name: str) -> _Distribution:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PackageNotFoundError(name)
        return _Distribution(name)

    monkeypatch.setattr(provider_metadata, "distribution", lookup)

    assert provider_metadata.distribution_summary("provider-later") is None
    assert provider_metadata.distribution_summary("provider-later") is not None
    assert calls == 2


@dataclass
class _NamelessDistribution:
    requested_name: str
    recorded_name: object

    @property
    def metadata(self) -> dict[str, object]:
        return {"Name": self.recorded_name}

    @property
    def version(self) -> str:
        return "1.0"


@pytest.mark.parametrize("recorded", [None, ""])
def test_distribution_summary_falls_back_to_requested_name_when_name_missing_or_empty(
    isolated_metadata_cache: None,
    monkeypatch: pytest.MonkeyPatch,
    recorded: object,
) -> None:
    del isolated_metadata_cache
    monkeypatch.setattr(
        provider_metadata,
        "distribution",
        lambda name: _NamelessDistribution(name, recorded),
    )
    summary = provider_metadata.distribution_summary("provider-alpha")
    assert summary is not None
    assert summary.name == "provider-alpha"
    assert summary.version == "1.0"


def test_distribution_summary_falls_back_when_name_is_non_str(
    isolated_metadata_cache: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del isolated_metadata_cache
    monkeypatch.setattr(
        provider_metadata,
        "distribution",
        lambda name: _NamelessDistribution(name, 42),
    )
    summary = provider_metadata.distribution_summary("provider-alpha")
    assert summary is not None
    assert summary.name == "provider-alpha"
