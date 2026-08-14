"""Operator-only construction of built-in catalog descriptors."""

from __future__ import annotations

from jacobian.catalog_build_context import CatalogBuildContext
from jacobian.catalog_build_resources import CatalogBuildResources
from jacobian.catalog_checkers import CatalogCheckerBuilder
from jacobian.catalog_foundations import CatalogFoundationBuilder
from jacobian.catalog_operations import CatalogOperationBuilder
from jacobian.catalog_resources import CatalogResourceBuilder
from jacobian.implementation import cached_package_digests
from jacobian.polytope import PolytopeService
from jacobian.provider_inventory import ProviderInventoryLoader


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
    resolver = ProviderInventoryLoader()
    try:
        with (
            context.checkers.policy_transaction(),
            context.store.transaction(),
            cached_package_digests(),
        ):
            runtimes = resolver.resolve()
            CatalogFoundationBuilder(context).bind(runtimes)
            graph = CatalogOperationBuilder(context).bind(
                polytope,
            )
            CatalogResourceBuilder(context).bind(graph)
            CatalogCheckerBuilder(context, resolver).bind(
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
