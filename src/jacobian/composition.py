"""Composition root for the runtime service graph and built-in portfolio."""

from __future__ import annotations

from pathlib import Path

from jacobian.installation.context import create_installation_context
from jacobian.portfolio import install_portfolio
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.config import RuntimeOptions
from jacobian.runtime.model import JacobianRuntime
from jacobian.runtime.portfolio import PortfolioResources
from jacobian.runtime.services import build_runtime_services


def compose_runtime(root: str | Path, options: RuntimeOptions) -> JacobianRuntime:
    """Construct one owned runtime and close partial state on failure."""

    core = bootstrap_services(root, options)
    services = None
    try:
        services = build_runtime_services(core)
        installation = create_installation_context(core, services, options)
        portfolio_resources = install_portfolio(installation, services)
        return JacobianRuntime(
            core,
            services,
            portfolio_resources,
            start_lean_warmup=lambda: _start_lean_warmup(portfolio_resources),
        )
    except BaseException as error:
        cleanup_failures: list[BaseException] = []
        if services is not None:
            try:
                services.close()
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)
        try:
            core.close()
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        if cleanup_failures:
            error.add_note(
                "runtime construction cleanup also failed: "
                + "; ".join(str(failure) for failure in cleanup_failures)
            )
        raise


def _start_lean_warmup(resources: PortfolioResources) -> None:
    if resources.lean is not None:
        resources.lean.start_mathlib_warmup()


__all__ = ["compose_runtime"]
