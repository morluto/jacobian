"""Assembly of the explicit built-in mathematical portfolio."""

from __future__ import annotations

from jacobian.implementation import cached_package_digests
from jacobian.portfolio.checker_binding import CheckerPortfolioBinder
from jacobian.portfolio.context import PortfolioContext
from jacobian.portfolio.core_binding import CoreOperationBinder
from jacobian.portfolio.foundation_binding import FoundationBinder
from jacobian.portfolio.provider_resolution import ProviderAvailabilityResolver
from jacobian.portfolio.resource_binding import ResourceOperationBinder
from jacobian.runtime.portfolio import PortfolioResources
from jacobian.runtime.services import RuntimeServices


def assemble_portfolio(
    context: PortfolioContext,
    services: RuntimeServices,
) -> PortfolioResources:
    """Assemble the complete built-in portfolio in its declared phase order.

    This function is the single composition boundary for the built-in
    portfolio. It owns both the ordering of binding phases and the durable
    transaction that couples operation/checker registration to store writes.
    The checker-policy lock is acquired before the SQLite transaction, as
    required by :class:`CheckerRegistry`, and package digests are cached for
    the duration of the same atomic assembly.
    """

    core = services.core
    if context.store is not core.store:
        raise ValueError("portfolio context must belong to runtime core")

    resources = PortfolioResources()
    resolver = ProviderAvailabilityResolver()
    try:
        with (
            core.checkers.policy_transaction(),
            core.store.transaction(),
            cached_package_digests(),
        ):
            runtimes = resolver.resolve()
            FoundationBinder(context).bind(
                core,
                runtimes,
            )
            graph = CoreOperationBinder(context).bind(
                services,
            )
            ResourceOperationBinder(context).bind(graph)
            CheckerPortfolioBinder(context, resolver).bind(
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
