"""Whole-portfolio installation coverage belongs to the integration lane."""

from jacobian.contracts.operations import ProviderAvailability
from jacobian.domains.polynomial_nullstellensatz.core import (
    MATERIALIZE_OPERATION_ID,
    VERIFY_OPERATION_ID,
)
from jacobian.domains.polynomial_nullstellensatz.singular import (
    PRODUCE_OPERATION_ID,
)
from jacobian.portfolio.builtin import build_builtin_portfolio
from jacobian.providers.singular_runtime import singular_provider_runtime
from jacobian.runtime.model import JacobianRuntime


def test_builtin_portfolio_installs_cleanly(
    fresh_complete_runtime: JacobianRuntime,
) -> None:
    singular_available = (
        singular_provider_runtime().availability
        is ProviderAvailability.AVAILABLE
    )
    expected_operation_ids = {
        operation_id
        for bundle in build_builtin_portfolio().components
        for operation_id in bundle.operation_ids
    }
    expected_operation_ids.update((MATERIALIZE_OPERATION_ID, VERIFY_OPERATION_ID))
    if singular_available:
        expected_operation_ids.add(PRODUCE_OPERATION_ID)
    installed_operation_ids = {
        descriptor.operation_id
        for descriptor in fresh_complete_runtime.core.operations.catalog().operations
    }
    assert expected_operation_ids <= installed_operation_ids
    assert (PRODUCE_OPERATION_ID in installed_operation_ids) is singular_available
