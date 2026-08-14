"""Complete built-in catalog compilation coverage for the integration lane."""

from jacobian.builtin_operation_modules import load_builtin_operation_modules
from jacobian.contracts.operations import ProviderAvailability
from jacobian.domains.polynomial_nullstellensatz.core import (
    MATERIALIZE_OPERATION_ID,
    VERIFY_OPERATION_ID,
)
from jacobian.domains.polynomial_nullstellensatz.singular import (
    PRODUCE_OPERATION_ID,
)
from jacobian.providers.singular_runtime import singular_provider_runtime
from jacobian.runtime.model import JacobianRuntime


def test_builtin_catalog_compiles_cleanly(
    fresh_complete_runtime: JacobianRuntime,
) -> None:
    singular_available = (
        singular_provider_runtime().availability is ProviderAvailability.AVAILABLE
    )
    expected_operation_ids = {
        operation_id
        for _module_name, operations, _checkers in load_builtin_operation_modules()
        for operation_id in (operation.operation_id for operation in operations)
    }
    expected_operation_ids.update((MATERIALIZE_OPERATION_ID, VERIFY_OPERATION_ID))
    if singular_available:
        expected_operation_ids.add(PRODUCE_OPERATION_ID)
    installed_operation_ids = {
        descriptor.operation_id
        for descriptor in fresh_complete_runtime.core.operations.snapshot().operations
    }
    assert expected_operation_ids <= installed_operation_ids
    assert (PRODUCE_OPERATION_ID in installed_operation_ids) is singular_available
