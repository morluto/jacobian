"""Ownership and lifecycle for one Jacobian application runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jacobian.polytope import PolytopeService
from jacobian.runtime.resources import RuntimeResources
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.runtime.execution import LazyControlPlane


class RuntimeClosedError(RuntimeError):
    """An operation requires a live Jacobian runtime."""


class JacobianRuntime:
    """Own one catalog-backed selected-operation execution runtime."""

    def __init__(
        self,
        core: RuntimeResources,
        verification: VerificationService | None = None,
        polytope: PolytopeService | None = None,
        *,
        control_plane: LazyControlPlane | None = None,
    ) -> None:
        if control_plane is None and (verification is None or polytope is None):
            raise TypeError(
                "JacobianRuntime requires verification and polytope or a control plane"
            )
        self._closed = False
        self.core = core
        self._verification = verification
        self._polytope = polytope
        self._control_plane = control_plane

    @property
    def verification(self) -> VerificationService:
        if self._verification is not None:
            return self._verification
        assert self._control_plane is not None
        return self._control_plane.verification

    @property
    def polytope(self) -> PolytopeService:
        if self._polytope is not None:
            return self._polytope
        assert self._control_plane is not None
        return self._control_plane.polytope

    def close(self) -> None:
        """Release every runtime-owned resource."""

        if self._closed:
            return
        failures: list[BaseException] = []
        for close in (self.core.close,):
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
