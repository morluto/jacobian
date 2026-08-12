"""Owned application opening for complete and scoped portfolio plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from jacobian.exact_domain_checkers import (
    ExactDomainCheckerInstallation,
    install_exact_domain_verification,
)
from jacobian.implementation import cached_package_digests
from jacobian.installation.context import (
    InstallationContext,
    create_installation_context,
)
from jacobian.operation_installation import InstalledDomainBundle
from jacobian.operations import DomainBundle
from jacobian.portfolio.application_plan import (
    ApplicationInstallPlan,
    InstallationReceipt,
    receipt_from_installed_bundles,
)
from jacobian.portfolio.assembler import install_portfolio
from jacobian.portfolio.builtin import build_builtin_portfolio
from jacobian.portfolio.domain_installation import DomainBundleInstaller
from jacobian.portfolio.model import PortfolioPlan
from jacobian.portfolio.result import PortfolioInstallation
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.config import RuntimeOptions
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.services import (
    CoreServices,
    RuntimeServices,
    build_runtime_services,
)


@dataclass(frozen=True, slots=True)
class OpenedApplication:
    """One owned runtime, its installation handles, and structural receipt."""

    plan: ApplicationInstallPlan
    runtime: JacobianRuntime
    installation: InstallationContext
    portfolio: PortfolioInstallation
    receipt: InstallationReceipt

    @property
    def core(self) -> CoreServices:
        """Return the foundational services owned by this application."""

        return self.runtime.core

    @property
    def services(self) -> RuntimeServices:
        """Return the retained mathematical services."""

        return self.runtime.services

    @property
    def bundles(self) -> Mapping[str, InstalledDomainBundle]:
        """Return installed domain bundles by domain ID."""

        return self.portfolio.domain_bundles

    def close(self) -> None:
        """Release all resources owned by this application."""

        self.runtime.close()

    def __enter__(self) -> OpenedApplication:
        self.runtime.__enter__()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def open_application(
    state_root: str | Path,
    plan: ApplicationInstallPlan,
) -> OpenedApplication:
    """Open one complete or scoped application described by ``plan``.

    Complete plans delegate to the production built-in portfolio assembler.
    Scoped plans select explicit built-in components and install them through
    ``DomainBundleInstaller``. Unknown IDs are rejected before opening state.
    """

    scoped_portfolio = _select_scoped_portfolio(plan)
    options = RuntimeOptions(checker_authority=plan.checker_authority)
    core = bootstrap_services(state_root, options)
    services: RuntimeServices | None = None
    portfolio: PortfolioInstallation | None = None
    try:
        services = build_runtime_services(core)
        installation = create_installation_context(core, services, options)
        if plan.kind == "complete":
            portfolio = install_portfolio(installation, services)
            receipt = receipt_from_installed_bundles(
                plan,
                portfolio.domain_bundles,
                checker_ids=_checker_ids(portfolio.exact_domain_checkers),
            )
        else:
            if scoped_portfolio is None:
                raise AssertionError("scoped plan selection was not constructed")
            portfolio, receipt = _install_scoped_application(
                installation,
                scoped_portfolio,
                plan,
            )
        runtime = JacobianRuntime(
            core,
            services,
            portfolio,
            start_lean_warmup=lambda: _start_lean_warmup(portfolio),
        )
        return OpenedApplication(
            plan=plan,
            runtime=runtime,
            installation=installation,
            portfolio=portfolio,
            receipt=receipt,
        )
    except BaseException as error:
        _close_partial_application(error, core, services, portfolio)
        raise


def _select_scoped_portfolio(
    plan: ApplicationInstallPlan,
) -> PortfolioPlan | None:
    if plan.kind == "complete":
        return None
    builtin = build_builtin_portfolio()
    unknown = tuple(
        domain_id
        for domain_id in plan.domain_ids
        if builtin.component_for(domain_id) is None
    )
    if unknown:
        raise ValueError("unknown scoped domain_id(s): " + ", ".join(unknown))
    return PortfolioPlan(
        components=tuple(
            component
            for domain_id in plan.domain_ids
            if (component := builtin.component_for(domain_id)) is not None
        )
    )


def _install_scoped_application(
    context: InstallationContext,
    scoped_portfolio: PortfolioPlan,
    plan: ApplicationInstallPlan,
) -> tuple[PortfolioInstallation, InstallationReceipt]:
    portfolio = PortfolioInstallation()
    try:
        with (
            context.checkers.policy_transaction(),
            context.store.transaction(),
            cached_package_digests(),
        ):
            result = DomainBundleInstaller(context).install(scoped_portfolio)
            missing = tuple(
                domain_id
                for domain_id in plan.domain_ids
                if domain_id not in result.installed
            )
            if missing:
                raise ValueError(
                    "scoped application installation omitted domain(s): "
                    + ", ".join(missing)
                )
            portfolio.domain_bundles = dict(result.installed)
            portfolio.portfolio_diagnostics = result.diagnostics
            portfolio.portfolio_outcomes = result.outcomes
            if plan.include_exact_verification:
                _install_scoped_exact_verification(
                    context,
                    scoped_portfolio,
                    portfolio,
                )
            receipt = receipt_from_installed_bundles(
                plan,
                portfolio.domain_bundles,
                checker_ids=_checker_ids(portfolio.exact_domain_checkers),
            )
    except BaseException as error:
        try:
            portfolio.close()
        except BaseException as cleanup_error:
            error.add_note(f"scoped portfolio cleanup also failed: {cleanup_error}")
        raise
    return portfolio, receipt


def _install_scoped_exact_verification(
    context: InstallationContext,
    scoped_portfolio: PortfolioPlan,
    portfolio: PortfolioInstallation,
) -> None:
    bundles = {
        component.domain_id: (
            component,
            portfolio.domain_bundles[component.domain_id],
        )
        for component in scoped_portfolio.components
        if isinstance(component, DomainBundle)
        and component.checker_declarations
        and component.domain_id in portfolio.domain_bundles
    }
    if not bundles:
        return
    adapters, portfolio.exact_domain_checkers = install_exact_domain_verification(
        context.store,
        context.schemas,
        context.artifacts,
        context.verification,
        context.checkers,
        bundles=bundles,
        authorize=context.authorizes_bundled_checkers,
    )
    for adapter in adapters:
        context.register_capability(adapter)


def _checker_ids(
    installation: ExactDomainCheckerInstallation | None,
) -> tuple[str, ...]:
    if installation is None:
        return ()
    return tuple(
        checker_id
        for checker_id in installation.checker_ids.values()
        if checker_id is not None
    )


def _start_lean_warmup(portfolio: PortfolioInstallation) -> None:
    if portfolio.lean is not None:
        portfolio.lean.start_mathlib_warmup()


def _close_partial_application(
    error: BaseException,
    core: CoreServices,
    services: RuntimeServices | None,
    portfolio: PortfolioInstallation | None,
) -> None:
    failures: list[BaseException] = []
    for resource in (services, portfolio, core):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as cleanup_error:
            failures.append(cleanup_error)
    if failures:
        error.add_note(
            "application construction cleanup also failed: "
            + "; ".join(str(failure) for failure in failures)
        )


__all__ = ["OpenedApplication", "open_application"]
