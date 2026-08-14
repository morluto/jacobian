from __future__ import annotations

from collections.abc import Callable

import pytest

import jacobian.maintained_backends as maintained_backends
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.provider_runtime import ProviderRuntimeError, known_provider_runtime


def test_required_backend_check_measures_each_pinned_math_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def probe(name: str) -> Callable[[], ProviderObservation]:
        def resolve() -> ProviderObservation:
            calls.append(name)
            return known_provider_runtime(name)

        return resolve

    monkeypatch.setattr(
        maintained_backends,
        "known_provider_runtime",
        lambda name: probe(name)(),
    )
    monkeypatch.setattr(
        maintained_backends,
        "python_flint_provider_runtime",
        probe("python-flint"),
    )
    monkeypatch.setattr(
        maintained_backends,
        "cvc5_provider_runtime",
        probe("cvc5"),
    )

    maintained_backends.require_maintained_math_backends()

    assert calls == [
        "jacobian.networkx",
        "jacobian.sympy",
        "jacobian.z3",
        "python-flint",
        "cvc5",
    ]


def test_required_backend_check_reports_every_broken_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = ProviderObservation(
        provider="jacobian.networkx",
        availability=ProviderAvailability.UNAVAILABLE,
        platform="test-platform",
        install_tier=ProviderInstallTier.T0,
        license_id="BSD-3-Clause",
        diagnostic="NetworkX is missing",
    )
    skewed = ProviderObservation(
        provider="jacobian.sympy",
        availability=ProviderAvailability.UNAVAILABLE,
        platform="test-platform",
        install_tier=ProviderInstallTier.T0,
        license_id="BSD-3-Clause",
        diagnostic="SymPy version does not match the pin",
    )
    observations = iter(
        (
            missing,
            skewed,
            known_provider_runtime("jacobian.z3"),
        )
    )
    monkeypatch.setattr(
        maintained_backends,
        "known_provider_runtime",
        lambda _name: next(observations),
    )
    available = known_provider_runtime("test.available")
    monkeypatch.setattr(
        maintained_backends,
        "python_flint_provider_runtime",
        lambda: available,
    )
    monkeypatch.setattr(
        maintained_backends,
        "cvc5_provider_runtime",
        lambda: available,
    )

    with pytest.raises(ProviderRuntimeError) as raised:
        maintained_backends.require_maintained_math_backends()

    assert str(raised.value) == (
        "required Python math backends are unavailable: "
        "jacobian.networkx: NetworkX is missing; "
        "jacobian.sympy: SymPy version does not match the pin"
    )
