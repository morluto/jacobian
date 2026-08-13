"""Whole-portfolio installation coverage belongs to the integration lane."""

from jacobian.contracts.capabilities import CapabilityProviderAvailability
from jacobian.domains.polynomial_nullstellensatz.core import (
    MATERIALIZE_CAPABILITY_ID,
    VERIFY_CAPABILITY_ID,
)
from jacobian.domains.polynomial_nullstellensatz.singular import (
    PRODUCE_CAPABILITY_ID,
)
from jacobian.portfolio.builtin import build_builtin_portfolio
from jacobian.providers.singular_runtime import singular_provider_runtime
from jacobian.runtime.model import JacobianRuntime


def test_builtin_portfolio_installs_cleanly(
    fresh_complete_runtime: JacobianRuntime,
) -> None:
    singular_available = (
        singular_provider_runtime().availability
        is CapabilityProviderAvailability.AVAILABLE
    )
    expected_capability_ids = {
        capability_id
        for bundle in build_builtin_portfolio().components
        for capability_id in bundle.capability_ids
    }
    expected_capability_ids.update((MATERIALIZE_CAPABILITY_ID, VERIFY_CAPABILITY_ID))
    if singular_available:
        expected_capability_ids.add(PRODUCE_CAPABILITY_ID)
    installed_capability_ids = {
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
    }
    assert expected_capability_ids <= installed_capability_ids
    assert (PRODUCE_CAPABILITY_ID in installed_capability_ids) is singular_available
