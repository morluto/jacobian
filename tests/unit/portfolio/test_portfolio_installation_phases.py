"""Direct seam tests for the explicit portfolio installation phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

from jacobian.contracts.capabilities import (
    CapabilityInstallTier,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
)
from jacobian.installation.context import InstallationContext
from jacobian.portfolio.core_installation import CoreApplicationInstaller
from jacobian.portfolio.foundation_installation import FoundationInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.portfolio.provider_resolution import (
    ProviderAvailabilityResolver,
    ProviderRuntimePlan,
)
from jacobian.portfolio.reference_installation import ReferenceLeanInstaller
from jacobian.portfolio.resource_installation import ResourceCapabilityInstaller
from jacobian.portfolio.result import PortfolioInstallation
from jacobian.runtime.config import CheckerAuthorityMode
from jacobian.runtime.services import ApplicationServices, CoreServices


def _unavailable_runtime(provider: str) -> CapabilityProviderRuntime:
    return CapabilityProviderRuntime(
        provider=provider,
        availability=CapabilityProviderAvailability.UNAVAILABLE,
        platform="test-platform",
        install_tier=CapabilityInstallTier.T0,
        license_id="MIT",
        diagnostic=f"{provider} is unavailable",
    )


def _unavailable_provider_plan() -> ProviderRuntimePlan:
    return ProviderRuntimePlan(
        cadical=_unavailable_runtime("cadical"),
        carcara=_unavailable_runtime("carcara"),
        cvc5=_unavailable_runtime("cvc5"),
        drat_trim=_unavailable_runtime("drat-trim"),
        python_flint=_unavailable_runtime("python-flint"),
        python_flint_hnf=_unavailable_runtime("python-flint-hnf"),
        sympy_polynomial_normalization=_unavailable_runtime(
            "sympy-polynomial-normalization"
        ),
    )


def test_foundation_optional_provider_phase_skips_unavailable_solvers() -> None:
    result = PortfolioInstallation()

    FoundationInstaller(
        cast(InstallationContext, object())
    ).install_optional_provider_components(
        cast(CoreServices, object()),
        result,
        _unavailable_provider_plan(),
    )

    assert result.cadical_runtime is not None
    assert (
        result.cadical_runtime.availability
        is CapabilityProviderAvailability.UNAVAILABLE
    )
    assert result.cvc5_runtime is not None
    assert (
        result.cvc5_runtime.availability is CapabilityProviderAvailability.UNAVAILABLE
    )


def test_core_domain_verification_phase_accepts_empty_bundle_result() -> None:
    result = PortfolioInstallation()
    CoreApplicationInstaller(
        cast(InstallationContext, object())
    ).install_domain_verification(result, PortfolioPlan(domain_bundles=()))

    assert result.exact_domain_checkers is None


def test_resource_phase_requires_core_graph_installation() -> None:
    installer = ResourceCapabilityInstaller(cast(InstallationContext, object()))

    with pytest.raises(
        RuntimeError,
        match="graph capabilities must precede resource installation",
    ):
        installer.install(PortfolioInstallation())


@dataclass
class _UnauthorizedContext:
    authorizes_bundled_checkers: bool = False
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE
    registered: list[object] = field(default_factory=list)

    def register_capability(self, adapter: object) -> None:
        self.registered.append(adapter)


def test_reference_phase_derives_authority_from_its_context() -> None:
    context = _UnauthorizedContext()
    application = cast(
        ApplicationServices,
        SimpleNamespace(core=SimpleNamespace(plugins=object())),
    )
    result = PortfolioInstallation()

    ReferenceLeanInstaller(
        cast(InstallationContext, context),
        cast(ProviderAvailabilityResolver, object()),
    ).install(application, result)

    assert result.references == {}
    assert result.lean_checkers == {}
    assert context.registered == []


def test_portfolio_close_releases_every_owned_lean_resource_once() -> None:
    closed: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    result = PortfolioInstallation()
    result.lean_declarations = cast(Any, Resource("declarations"))
    result.lean_exploration = cast(
        Any,
        SimpleNamespace(repl=Resource("exploration")),
    )
    result.lean = cast(Any, Resource("verification"))

    result.close()
    result.close()

    assert closed == ["declarations", "exploration", "verification"]
