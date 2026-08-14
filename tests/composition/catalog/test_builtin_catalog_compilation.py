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
from jacobian.runtime.selected_families import (
    selected_family_specs,
    selected_operation_origin,
)


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


def test_compiled_family_ids_match_selected_operation_ids(
    fresh_complete_runtime: JacobianRuntime,
) -> None:
    installed_operation_ids = {
        descriptor.operation_id
        for descriptor in fresh_complete_runtime.core.operations.snapshot().operations
    }
    selected_ids = {
        operation_id
        for spec in selected_family_specs()
        for operation_id in spec.operation_ids
    }
    compiled_family_ids = selected_ids & installed_operation_ids
    assert compiled_family_ids
    for spec in selected_family_specs():
        for operation_id in spec.operation_ids & installed_operation_ids:
            assert selected_operation_origin(operation_id) == spec.origin
    for operation_id in installed_operation_ids - selected_ids:
        assert selected_operation_origin(operation_id) is None
