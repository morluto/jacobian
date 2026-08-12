"""Direct seam tests for the explicit portfolio installation phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderDigestKind,
    CapabilityProviderRuntime,
)
from jacobian.installation.context import InstallationContext
from jacobian.portfolio import foundation_installation
from jacobian.portfolio.checker_installation import CheckerPortfolioInstaller
from jacobian.portfolio.core_installation import CoreApplicationInstaller
from jacobian.portfolio.foundation_installation import FoundationInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.portfolio.provider_resolution import (
    ProviderAvailabilityResolver,
    ProviderRuntimePlan,
)
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.runtime.portfolio import PortfolioResources
from jacobian.runtime.services import CoreServices, RuntimeServices


def _unavailable_runtime(provider: str) -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider=provider,
        availability=CapabilityProviderAvailability.UNAVAILABLE,
        platform="test-platform",
        install_tier=CapabilityInstallTier.T0,
        license_id="MIT",
        diagnostic=f"{provider} is unavailable",
    )


def _available_runtime(provider: str) -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider=provider,
        availability=CapabilityProviderAvailability.AVAILABLE,
        version="1",
        digest="sha256:" + "0" * 64,
        digest_kind=CapabilityProviderDigestKind.SOURCE_TREE,
        platform="test-platform",
        install_tier=CapabilityInstallTier.T0,
        license_id="MIT",
    )


def _provider_plan_with_unavailable_external_solvers() -> ProviderRuntimePlan:
    return ProviderRuntimePlan(
        cadical=_unavailable_runtime("cadical"),
        carcara=_unavailable_runtime("carcara"),
        cvc5=_available_runtime("cvc5"),
        drat_trim=_unavailable_runtime("drat-trim"),
        sympy_polynomial_normalization=_available_runtime(
            "sympy-polynomial-normalization"
        ),
    )


def test_foundation_solver_phase_skips_unavailable_external_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    context = SimpleNamespace(register_capability=registered.append)
    adapter = object()
    monkeypatch.setattr(
        foundation_installation,
        "install_cvc5_capability",
        lambda _smt, _runtime: adapter,
    )

    FoundationInstaller(cast(InstallationContext, context)).install_solver_components(
        cast(CoreServices, SimpleNamespace(smt=object())),
        _provider_plan_with_unavailable_external_solvers(),
    )

    assert (
        _provider_plan_with_unavailable_external_solvers().cadical.availability
        is CapabilityProviderAvailability.UNAVAILABLE
    )
    assert registered == [adapter]


def test_core_domain_verification_phase_accepts_empty_bundle_result() -> None:
    assert (
        CoreApplicationInstaller(
            cast(InstallationContext, object())
        ).install_domain_verification(
            SimpleNamespace(installed={}), PortfolioPlan(components=())
        )
        is None
    )


@dataclass
class _UnauthorizedContext:
    authorizes_bundled_checkers: bool = False
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE
    registered: list[object] = field(default_factory=list)

    def register_capability(self, adapter: object) -> None:
        self.registered.append(adapter)


def test_checker_phase_derives_authority_from_its_context() -> None:
    context = _UnauthorizedContext()
    application = cast(
        RuntimeServices,
        SimpleNamespace(
            core=SimpleNamespace(),
        ),
    )
    CheckerPortfolioInstaller(
        cast(InstallationContext, context),
        cast(ProviderAvailabilityResolver, object()),
    ).install(application, PortfolioResources())

    assert context.registered == []


def test_portfolio_close_releases_every_owned_lean_resource_once() -> None:
    closed: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    result = PortfolioResources()
    result.lean_declarations = cast(Any, Resource("declarations"))
    result.lean_exploration = cast(
        Any,
        SimpleNamespace(repl=Resource("exploration")),
    )
    result.lean = cast(Any, Resource("verification"))

    result.close()
    result.close()

    assert closed == ["declarations", "exploration", "verification"]


def test_portfolio_close_continues_after_keyboard_interrupt() -> None:
    closed: list[str] = []

    class InterruptingResource:
        def close(self) -> None:
            closed.append("declarations")
            raise KeyboardInterrupt("declarations close interrupted")

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    result = PortfolioResources()
    result.lean_declarations = cast(Any, InterruptingResource())
    result.lean_exploration = cast(
        Any,
        SimpleNamespace(repl=Resource("exploration")),
    )
    result.lean = cast(Any, Resource("verification"))

    with pytest.raises(
        BaseExceptionGroup, match="portfolio resources failed to close"
    ) as exc:
        result.close()

    assert closed == ["declarations", "exploration", "verification"]
    assert [str(failure) for failure in exc.value.exceptions] == [
        "declarations close interrupted",
    ]
