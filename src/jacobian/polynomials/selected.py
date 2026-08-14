"""Lazy binding for the selected polynomial operation family."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import OperationCatalog
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.selected_operation_bindings import SelectedOperationBinding
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

SELECTED_POLYNOMIAL_OPERATION_IDS = frozenset(
    {
        "polynomial.expression.normalize",
        "polynomial.expression_normalization.verify",
        "polynomial.interval.positivity.decide",
        "polynomial.interval.positivity.verify",
        "polynomial.interval.enclose",
        "polynomial.interval.enclosure.verify",
        "polynomial.map.evaluate",
        "polynomial.map.compute_jacobian",
        "polynomial.map.keller_condition.verify",
        "polynomial.map.collision_witness",
        "polynomial.map.collision.search",
        "polynomial.map.collision.verify",
        "polynomial.map.collision_evidence.verify",
        "polynomial.map.inverse.refute_by_collision",
        "polynomial.identity.verify",
        "polynomial.rational_function.identity.verify",
        "polynomial.map.inverse.candidate_synthesize",
        "polynomial.map.inverse.verify",
        "polynomial.system.solution.verify",
        "polynomial.system.rational_solution.search",
        "polynomial.jacobian_degree_slice.system.materialize",
        "polynomial.nullstellensatz.infeasibility_certificate.compute",
        "polynomial.nullstellensatz.infeasibility_certificate.verify",
    }
)


def bind_selected_polynomial_operation(
    operation_id: str,
    descriptor: OperationDescriptor,
    *,
    binder: OperationBinder,
    verification: VerificationService,
    checkers: CheckerRegistry,
    polynomial_expressions: PolynomialExpressionArtifactService,
    catalog: OperationCatalog,
    store: ArtifactRepository,
    schemas: SchemaRegistry,
) -> SelectedOperationBinding | None:
    """Bind one selected polynomial operation and its exact dependencies."""

    if operation_id not in SELECTED_POLYNOMIAL_OPERATION_IDS:
        return None
    if operation_id == "polynomial.expression.normalize":
        from jacobian.providers.sympy_runtime import (
            sympy_polynomial_normalization_provider_runtime,
        )
        from jacobian.sympy_polynomial_normalization import (
            bind_sympy_polynomial_normalization,
        )

        adapter = bind_sympy_polynomial_normalization(
            polynomial_expressions,
            sympy_polynomial_normalization_provider_runtime(),
        )
        return SelectedOperationBinding(adapter)
    if operation_id == "polynomial.expression_normalization.verify":
        from jacobian.polynomial_expression_operations import (
            bind_selected_polynomial_expression_checker,
        )

        return SelectedOperationBinding(
            bind_selected_polynomial_expression_checker(
                store,
                schemas,
                binder.artifacts,
                polynomial_expressions,
                verification,
                checkers,
                catalog,
            )
        )
    if operation_id in {
        "polynomial.interval.positivity.decide",
        "polynomial.interval.positivity.verify",
    }:
        from jacobian.polynomial_positivity_operations import (
            bind_selected_polynomial_positivity_operation,
        )

        positivity_adapter = bind_selected_polynomial_positivity_operation(
            operation_id,
            store,
            schemas,
            binder.artifacts,
            verification,
            checkers,
            catalog,
        )
        return (
            None
            if positivity_adapter is None
            else SelectedOperationBinding(positivity_adapter)
        )
    if operation_id in {
        "polynomial.interval.enclose",
        "polynomial.interval.enclosure.verify",
    }:
        from jacobian.polynomial_interval_operations import (
            bind_selected_polynomial_interval_operation,
        )

        interval_adapter = bind_selected_polynomial_interval_operation(
            operation_id,
            store,
            schemas,
            binder.artifacts,
            verification,
            checkers,
            catalog,
        )
        return (
            None
            if interval_adapter is None
            else SelectedOperationBinding(interval_adapter)
        )
    from jacobian.polynomials.operation_build import (
        bind_selected_polynomial_operation as bind_operation,
    )

    polynomial_adapter = bind_operation(
        operation_id,
        store,
        schemas,
        binder.artifacts,
        verification,
        checkers,
        catalog,
    )
    if polynomial_adapter is not None:
        return SelectedOperationBinding(polynomial_adapter)
    from jacobian.polynomial_system_operations import (
        bind_selected_polynomial_system_operation,
    )

    system_adapter = bind_selected_polynomial_system_operation(
        operation_id,
        store,
        schemas,
        binder.artifacts,
        verification,
        checkers,
        catalog,
    )
    if system_adapter is not None:
        return SelectedOperationBinding(system_adapter)
    if operation_id == ("polynomial.nullstellensatz.infeasibility_certificate.compute"):
        from jacobian.domains.polynomial_nullstellensatz.singular import (
            bind_selected_singular_producer,
        )
        from jacobian.providers.singular_runtime import singular_provider_runtime

        return SelectedOperationBinding(
            bind_selected_singular_producer(
                store,
                schemas,
                binder.artifacts,
                singular_provider_runtime(),
            )
        )
    from jacobian.domains.polynomial_nullstellensatz.core import (
        bind_selected_nullstellensatz_operation,
    )
    from jacobian.provider_runtime import known_provider_runtime

    provider_runtime = known_provider_runtime(
        "jacobian.nullstellensatz-core",
        features=(
            "normalized-jacobian-degree-slice",
            "rabinowitsch-chart-cover",
            "independent-exact-replay",
        ),
    )
    binding = catalog.checker_binding(operation_id)
    if binding is not None:
        provider_runtime = provider_runtime.model_copy(
            update={"checker_ids": (binding.checker_id,)}
        )
    nullstellensatz_adapter = bind_selected_nullstellensatz_operation(
        operation_id,
        store,
        schemas,
        binder.artifacts,
        verification,
        checkers,
        catalog,
        provider_runtime,
    )
    return (
        None
        if nullstellensatz_adapter is None
        else SelectedOperationBinding(nullstellensatz_adapter)
    )


__all__ = [
    "SELECTED_POLYNOMIAL_OPERATION_IDS",
    "bind_selected_polynomial_operation",
]
