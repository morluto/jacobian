"""Ownership and lifecycle for one Jacobian application runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from jacobian.runtime.services import CoreServices, RuntimeServices


class RuntimeClosedError(RuntimeError):
    """An operation requires a live Jacobian runtime."""


class _PortfolioLifecycle(Protocol):
    def close(self) -> None: ...


class JacobianRuntime:
    """Own the explicit service graph and installed portfolio for one store."""

    def __init__(
        self,
        core: CoreServices,
        services: RuntimeServices,
        portfolio: _PortfolioLifecycle,
        start_lean_warmup: Callable[[], None],
    ) -> None:
        if services.core is not core:
            raise ValueError("runtime services must belong to the supplied core")
        self._closed = False
        self.core = core
        self.services = services
        self.portfolio = portfolio
        self._start_lean_warmup = start_lean_warmup

    def start_lean_warmup(self) -> None:
        """Warm the installed Lean backend when the portfolio provides one."""

        self._start_lean_warmup()

    def close(self) -> None:
        """Release every runtime-owned resource."""

        if self._closed:
            return
        failures: list[BaseException] = []
        for close in (
            self.services.close,
            self.portfolio.close,
            self.core.close,
        ):
            try:
                close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            exception_failures = [
                failure for failure in failures if isinstance(failure, Exception)
            ]
            if len(exception_failures) == len(failures):
                raise ExceptionGroup(
                    "runtime resources failed to close", exception_failures
                )
            raise BaseExceptionGroup("runtime resources failed to close", failures)
        self._closed = True

    def __enter__(self) -> JacobianRuntime:
        if self._closed:
            raise RuntimeClosedError("Jacobian runtime is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["JacobianRuntime", "RuntimeClosedError"]
