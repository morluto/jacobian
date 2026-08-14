"""Lazy binding for the selected polynomial operation family."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jacobian.contracts.operations import OperationDescriptor
from jacobian.operation_binding import OperationBinder
from jacobian.operation_catalog import OperationCatalog
from jacobian.polynomial_expressions import PolynomialExpressionArtifactService
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry
from jacobian.selected_operation_bindings import SelectedOperationBinding
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.catalog.build import CatalogBuildContext

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
    "install_selected_polynomial_catalog",
]


def install_selected_polynomial_catalog(
    context: CatalogBuildContext,
    *,
    polytope: object | None = None,
    resources: object | None = None,
) -> None:
    """Compile polynomial map, system, search, Nullstellensatz, and interval ops."""

    del polytope, resources
    _install_polynomial_expressions(context)
    _install_nullstellensatz(context)
    _install_polynomial_maps_and_systems(context)
    _install_polynomial_interval_and_positivity(context)


def _install_polynomial_expressions(context: CatalogBuildContext) -> None:
    from jacobian.polynomial_expression_operations import (
        install_polynomial_expression_checker,
    )
    from jacobian.providers.sympy_runtime import (
        sympy_polynomial_normalization_provider_runtime,
    )
    from jacobian.sympy_polynomial_normalization import (
        bind_sympy_polynomial_normalization,
    )

    verification_adapter, _ = install_polynomial_expression_checker(
        context.store,
        context.schemas,
        context.artifacts,
        context.polynomial_expressions,
        context.verification,
        context.checkers,
        authorize_checker=context.authorize_bundled_checkers,
    )
    if verification_adapter is not None:
        context.register_operation(verification_adapter)
    context.register_operation(
        bind_sympy_polynomial_normalization(
            context.polynomial_expressions,
            sympy_polynomial_normalization_provider_runtime(),
        )
    )


def _install_nullstellensatz(context: CatalogBuildContext) -> None:
    from jacobian.contracts.operations import ProviderAvailability
    from jacobian.domains.polynomial_nullstellensatz.core import (
        install_nullstellensatz_core,
    )
    from jacobian.domains.polynomial_nullstellensatz.singular import (
        install_singular_producer,
    )
    from jacobian.provider_runtime import known_provider_runtime
    from jacobian.providers.singular_runtime import singular_provider_runtime

    core_runtime = known_provider_runtime(
        "jacobian.nullstellensatz-core",
        features=(
            "normalized-jacobian-degree-slice",
            "rabinowitsch-chart-cover",
            "independent-exact-replay",
        ),
    )
    core = install_nullstellensatz_core(context, core_runtime)
    for adapter in core.adapters:
        context.register_operation(adapter)
    singular_runtime = singular_provider_runtime()
    if singular_runtime.availability is not ProviderAvailability.AVAILABLE:
        return
    singular = install_singular_producer(context, core, singular_runtime)
    for adapter in singular.adapters:
        context.register_operation(adapter)


def _install_polynomial_maps_and_systems(context: CatalogBuildContext) -> None:
    from jacobian.polynomial_system_operations import (
        install_polynomial_system_operations,
    )
    from jacobian.polynomial_system_search import PolynomialSystemRationalSearchAdapter
    from jacobian.polynomials import build_polynomial_operations

    polynomial_adapters, _ = build_polynomial_operations(
        context.store,
        context.schemas,
        context.artifacts,
        context.verification,
        context.checkers,
        authorize_checker=context.authorize_bundled_checkers,
    )
    for polynomial_adapter in polynomial_adapters:
        context.register_operation(polynomial_adapter)
    polynomial_system_adapter, polynomial_system = install_polynomial_system_operations(
        context.store,
        context.schemas,
        context.artifacts,
        context.verification,
        context.checkers,
        authorize_checker=context.authorize_bundled_checkers,
    )
    if polynomial_system_adapter is not None:
        context.register_operation(polynomial_system_adapter)
    context.register_operation(
        PolynomialSystemRationalSearchAdapter(context.artifacts, polynomial_system)
    )


def _install_polynomial_interval_and_positivity(context: CatalogBuildContext) -> None:
    from jacobian.polynomial_interval_operations import (
        install_polynomial_interval_operations,
    )
    from jacobian.polynomial_positivity_operations import (
        install_polynomial_positivity_operations,
    )

    interval_adapters, _ = install_polynomial_interval_operations(
        context.store,
        context.schemas,
        context.artifacts,
        context.verification,
        context.checkers,
        authorize_checker=context.authorize_bundled_checkers,
    )
    for interval_adapter in interval_adapters:
        if interval_adapter is not None:
            context.register_operation(interval_adapter)
    positivity_adapters, _ = install_polynomial_positivity_operations(
        context.store,
        context.schemas,
        context.artifacts,
        context.verification,
        context.checkers,
        authorize_checker=context.authorize_bundled_checkers,
    )
    for positivity_adapter in positivity_adapters:
        if positivity_adapter is not None:
            context.register_operation(positivity_adapter)
