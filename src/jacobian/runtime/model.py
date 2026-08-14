"""Ownership and lifecycle for one Jacobian application runtime."""

from __future__ import annotations

from collections.abc import Callable

from jacobian.polytope import PolytopeService
from jacobian.runtime.resources import RuntimeResources
from jacobian.verification.service import VerificationService


class RuntimeClosedError(RuntimeError):
    """An operation requires a live Jacobian runtime."""


class JacobianRuntime:
    """Own one catalog-backed selected-operation execution runtime."""

    def __init__(
        self,
        core: RuntimeResources,
        verification: VerificationService,
        polytope: PolytopeService,
        *,
        close_resources: Callable[[], None] | None = None,
    ) -> None:
        self._closed = False
        self.core = core
        self.verification = verification
        self.polytope = polytope
        self._close_resources = close_resources or (lambda: None)

    def close(self) -> None:
        """Release every runtime-owned resource."""

        if self._closed:
            return
        failures: list[BaseException] = []
        for close in (
            self._close_resources,
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
