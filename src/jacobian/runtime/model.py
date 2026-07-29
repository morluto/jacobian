"""Ownership and lifecycle for one Jacobian application runtime."""

from __future__ import annotations

from pathlib import Path

from jacobian.installation.context import create_installation_context
from jacobian.runtime.bootstrap import bootstrap_services
from jacobian.runtime.config import RuntimeOptions
from jacobian.runtime.services import build_application_services


class RuntimeClosedError(RuntimeError):
    """An operation requires a live Jacobian runtime."""


class JacobianRuntime:
    """Own the explicit service graph and installed portfolio for one store."""

    def __init__(self, root: str | Path, options: RuntimeOptions) -> None:
        self._closed = False
        self.core = bootstrap_services(root, options)
        try:
            from jacobian.portfolio import install_portfolio

            self.services = build_application_services(self.core)
            installation = create_installation_context(
                self.core,
                self.services,
                options,
            )
            self.portfolio = install_portfolio(
                installation,
                self.services,
                capability_adapter_entrypoints=options.capability_adapter_entrypoints,
            )
        except BaseException:
            self.core.close()
            self._closed = True
            raise

    def close(self) -> None:
        """Release every runtime-owned resource."""

        if self._closed:
            return
        self.services.close()
        self.core.close()
        self._closed = True

    def __enter__(self) -> JacobianRuntime:
        if self._closed:
            raise RuntimeClosedError("Jacobian runtime is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["JacobianRuntime", "RuntimeClosedError"]
