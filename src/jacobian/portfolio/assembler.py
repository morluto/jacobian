"""Ordered installation of the explicit built-in mathematical portfolio."""

from __future__ import annotations

from jacobian.implementation import cached_package_digests
from jacobian.installation.context import InstallationContext
from jacobian.portfolio.core_installation import CoreApplicationInstaller
from jacobian.portfolio.foundation_installation import FoundationInstaller
from jacobian.portfolio.provider_resolution import ProviderAvailabilityResolver
from jacobian.portfolio.reference_installation import ReferenceLeanInstaller
from jacobian.portfolio.resource_installation import ResourceCapabilityInstaller
from jacobian.portfolio.result import PortfolioInstallation
from jacobian.runtime.services import ApplicationServices


def install_portfolio(
    context: InstallationContext,
    application: ApplicationServices,
    *,
    capability_adapter_entrypoints: tuple[str, ...] = (),
) -> PortfolioInstallation:
    """Install the complete portfolio in its declared phase order.

    This function is the single composition boundary for the built-in
    portfolio.  It owns both the ordering of phase installers and the durable
    transaction that couples capability/checker registration to store writes.
    The checker-policy lock is acquired before the SQLite transaction, as
    required by :class:`CheckerRegistry`, and package digests are cached for
    the duration of the same atomic installation.
    """

    core = application.core
    if context.store is not core.store:
        raise ValueError("installation context must belong to application core")

    result = PortfolioInstallation()
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
                result,
                runtimes,
            )
            CoreApplicationInstaller(context).install(
                application,
                result,
            )
            ResourceCapabilityInstaller(context).install(result)
            ReferenceLeanInstaller(context, resolver).install(
                application,
                result,
                capability_adapter_entrypoints=capability_adapter_entrypoints,
            )
    except BaseException as exc:
        try:
            result.close()
        except BaseException as cleanup_exc:
            exc.add_note(f"partial portfolio cleanup also failed: {cleanup_exc}")
        raise
    return result
