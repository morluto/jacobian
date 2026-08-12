"""Ordered installation of the explicit built-in mathematical portfolio."""

from __future__ import annotations

from jacobian.implementation import cached_package_digests
from jacobian.installation.context import InstallationContext
from jacobian.portfolio.checker_installation import CheckerPortfolioInstaller
from jacobian.portfolio.core_installation import CoreApplicationInstaller
from jacobian.portfolio.foundation_installation import FoundationInstaller
from jacobian.portfolio.provider_resolution import ProviderAvailabilityResolver
from jacobian.portfolio.resource_installation import ResourceCapabilityInstaller
from jacobian.runtime.portfolio import PortfolioResources
from jacobian.runtime.services import RuntimeServices


def install_portfolio(
    context: InstallationContext,
    services: RuntimeServices,
) -> PortfolioResources:
    """Install the complete portfolio in its declared phase order.

    This function is the single composition boundary for the built-in
    portfolio.  It owns both the ordering of phase installers and the durable
    transaction that couples capability/checker registration to store writes.
    The checker-policy lock is acquired before the SQLite transaction, as
    required by :class:`CheckerRegistry`, and package digests are cached for
    the duration of the same atomic installation.
    """

    core = services.core
    if context.store is not core.store:
        raise ValueError("installation context must belong to runtime core")

    resources = PortfolioResources()
    resolver = ProviderAvailabilityResolver()
    try:
        with (
            core.checkers.policy_transaction(),
            core.store.transaction(),
            cached_package_digests(),
        ):
            runtimes = resolver.resolve()
            FoundationInstaller(context).install(
                core,
                runtimes,
            )
            graph = CoreApplicationInstaller(context).install(
                services,
            )
            ResourceCapabilityInstaller(context).install(graph)
            CheckerPortfolioInstaller(context, resolver).install(
                services,
                resources,
            )
    except BaseException as exc:
        try:
            resources.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"partial portfolio cleanup also failed: {cleanup_exc}")
        raise
    return resources
