"""Operator-only construction of built-in catalog descriptors."""

from __future__ import annotations

from jacobian.catalog_build_context import CatalogBuildContext
from jacobian.catalog_build_resources import CatalogBuildResources
from jacobian.catalog_checkers import CatalogCheckerBuilder
from jacobian.catalog_foundations import bind_catalog_foundations
from jacobian.catalog_operations import CatalogOperationBuilder
from jacobian.catalog_resources import CatalogResourceBuilder
from jacobian.implementation import cached_package_digests
from jacobian.maintained_backends import require_maintained_math_backends
from jacobian.polytope import PolytopeService
from jacobian.providers.external_solver_runtime import (
    cadical_provider_runtime,
    carcara_provider_runtime,
    cvc5_provider_runtime,
    drat_trim_provider_runtime,
)
from jacobian.providers.sympy_runtime import (
    sympy_polynomial_normalization_provider_runtime,
)


def build_catalog_operations(
    context: CatalogBuildContext,
    polytope: PolytopeService,
) -> CatalogBuildResources:
    """Build every descriptor and checker binding in deterministic phase order.

    This function is the single build boundary for the built-in catalog. It
    owns both the ordering of compilation phases and the durable
    transaction that couples operation/checker registration to store writes.
    The checker-policy lock is acquired before the SQLite transaction, as
    required by :class:`CheckerRegistry`, and package digests are cached for
    the duration of the same atomic assembly.
    """

    resources = CatalogBuildResources()
    try:
        with (
            context.checkers.policy_transaction(),
            context.store.transaction(),
            cached_package_digests(),
        ):
            require_maintained_math_backends()
            bind_catalog_foundations(
                context,
                cadical=cadical_provider_runtime(),
                carcara=carcara_provider_runtime(),
                cvc5=cvc5_provider_runtime(),
                drat_trim=drat_trim_provider_runtime(),
                sympy_normalization=(sympy_polynomial_normalization_provider_runtime()),
            )
            graph = CatalogOperationBuilder(context).bind(
                polytope,
            )
            CatalogResourceBuilder(context).bind(graph)
            CatalogCheckerBuilder(context).bind(
                polytope,
                resources,
            )
    except BaseException as exc:
        try:
            resources.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"partial catalog cleanup also failed: {cleanup_exc}")
        raise
    return resources
