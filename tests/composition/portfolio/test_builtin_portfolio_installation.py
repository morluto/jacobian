"""Whole-portfolio installation coverage belongs to the integration lane."""

from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.domains.polynomial_nullstellensatz.bundle import SINGULAR_DOMAIN_ID
from jacobian.portfolio import build_builtin_portfolio
from jacobian.portfolio.result import (
    PROVIDER_UNAVAILABLE,
    BundleInstallationStatus,
)
from jacobian.providers.singular_runtime import singular_provider_runtime
from jacobian.runtime.model import JacobianRuntime

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "WIRING"


def test_builtin_portfolio_installs_cleanly(
    fresh_complete_runtime: JacobianRuntime,
) -> None:
    installation = fresh_complete_runtime.portfolio

    expected_domain_ids = set(build_builtin_portfolio().domain_ids)
    singular_available = (
        singular_provider_runtime().availability
        is CapabilityProviderAvailability.AVAILABLE
    )
    if singular_available:
        assert installation.portfolio_diagnostics == ()
        assert set(installation.domain_bundles) == expected_domain_ids
        assert all(
            outcome.status is BundleInstallationStatus.INSTALLED
            for outcome in installation.portfolio_outcomes
        )
    else:
        assert set(installation.domain_bundles) == expected_domain_ids - {
            SINGULAR_DOMAIN_ID
        }
        assert len(installation.portfolio_diagnostics) == 1
        diagnostic = installation.portfolio_diagnostics[0]
        assert diagnostic.component_id == SINGULAR_DOMAIN_ID
        assert diagnostic.code == PROVIDER_UNAVAILABLE
        singular_outcome = next(
            outcome
            for outcome in installation.portfolio_outcomes
            if outcome.domain_id == SINGULAR_DOMAIN_ID
        )
        assert (
            singular_outcome.status
            is BundleInstallationStatus.SKIPPED_PROVIDER_UNAVAILABLE
        )
        assert all(
            outcome.status is BundleInstallationStatus.INSTALLED
            for outcome in installation.portfolio_outcomes
            if outcome.domain_id != SINGULAR_DOMAIN_ID
        )
    expected_capability_ids = {
        operation.capability_id
        for bundle in build_builtin_portfolio().domain_bundles
        if singular_available or bundle.domain_id != SINGULAR_DOMAIN_ID
        for operation in bundle.capabilities
    }
    installed_capability_ids = {
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
    }
    assert expected_capability_ids <= installed_capability_ids
