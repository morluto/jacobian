"""Direct tests for the provider-availability resolution phase."""

from __future__ import annotations

from collections.abc import Callable

import pytest

import jacobian.provider_inventory as provider_resolution
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.provider_inventory import ProviderInventoryLoader
from jacobian.provider_runtime import ProviderRuntimeError, known_provider_runtime


def test_resolve_builds_one_typed_plan_from_each_declared_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def probe(name: str) -> Callable[[], ProviderObservation]:
        def resolve() -> ProviderObservation:
            calls.append(name)
            return known_provider_runtime(name)

        return resolve

    names = (
        "cvc5",
        "sympy_polynomial_normalization",
        "cadical",
        "carcara",
        "drat_trim",
    )
    for name in names:
        monkeypatch.setattr(
            provider_resolution,
            f"{name}_provider_runtime",
            probe(name),
        )
    monkeypatch.setattr(
        provider_resolution,
        "python_flint_provider_runtime",
        lambda: known_provider_runtime("python_flint"),
    )

    plan = ProviderInventoryLoader().resolve()

    assert calls == list(names)
    assert tuple(getattr(plan, name).provider for name in names) == names


def test_resolve_rejects_a_missing_packaged_python_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = ProviderObservation(
        provider="jacobian.networkx",
        availability=ProviderAvailability.UNAVAILABLE,
        platform="test-platform",
        install_tier=ProviderInstallTier.T0,
        license_id="BSD-3-Clause",
        diagnostic="NetworkX is missing",
    )
    monkeypatch.setattr(
        provider_resolution,
        "known_provider_runtime",
        lambda name: (
            unavailable if name == "jacobian.networkx" else known_provider_runtime(name)
        ),
    )

    with pytest.raises(
        ProviderRuntimeError,
        match=r"required Python providers are unavailable: jacobian\.networkx",
    ):
        ProviderInventoryLoader().resolve()


def test_packaged_backend_failure_reports_every_broken_provider() -> None:
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

    with pytest.raises(ProviderRuntimeError) as raised:
        provider_resolution._require_packaged_python_backends((missing, skewed))

    assert str(raised.value) == (
        "required Python providers are unavailable: "
        "jacobian.networkx: NetworkX is missing; "
        "jacobian.sympy: SymPy version does not match the pin"
    )


def test_lean_resolution_preserves_installed_checker_profile_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def resolve_lean(
        *,
        profiles: object,
        checker_ids: tuple[str, ...],
    ) -> ProviderObservation:
        observed["profiles"] = profiles
        observed["checker_ids"] = checker_ids
        return known_provider_runtime("lean")

    monkeypatch.setattr(provider_resolution, "lean_provider_runtime", resolve_lean)
    profiles = {"mathlib": {"semantics_uri": "semantics://lean"}}

    runtime = ProviderInventoryLoader().resolve_lean(
        profiles=profiles,
        checker_ids=("lean.mathlib",),
    )

    assert runtime.provider == "lean"
    assert observed == {
        "profiles": profiles,
        "checker_ids": ("lean.mathlib",),
    }


def test_lean_frontend_resolution_uses_dedicated_health_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = known_provider_runtime("lean-frontend")
    monkeypatch.setattr(
        provider_resolution,
        "lean_frontend_provider_runtime",
        lambda: expected,
    )

    runtime = ProviderInventoryLoader().resolve_lean_frontend()

    assert runtime is expected
