"""Ownership and lifecycle for one Jacobian application runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jacobian.catalog.collector import CatalogOperationCollector
from jacobian.operation_dispatcher import OperationDispatcher
from jacobian.polytope import PolytopeService
from jacobian.runtime.resources import RuntimeResources
from jacobian.verification.service import VerificationService

if TYPE_CHECKING:
    from jacobian.runtime.execution import LazyControlPlane


@dataclass(slots=True)
class InlineServingResources:
    """Dispatcher-only resources for packaged inline operations."""

    operations: OperationDispatcher

    def close(self) -> None:
        self.operations.close()


type RuntimeCore = RuntimeResources | InlineServingResources


class RuntimeClosedError(RuntimeError):
    """An operation requires a live Jacobian runtime."""


class JacobianRuntime:
    """Own one catalog-backed selected-operation execution runtime."""

    def __init__(
        self,
        core: RuntimeCore,
        verification: VerificationService | None = None,
        polytope: PolytopeService | None = None,
        *,
        control_plane: LazyControlPlane | None = None,
        inline_serving: bool = False,
    ) -> None:
        if (
            not inline_serving
            and control_plane is None
            and (verification is None or polytope is None)
        ):
            raise TypeError(
                "JacobianRuntime requires verification and polytope or a control plane"
            )
        self._closed = False
        self.core = core
        self._verification = verification
        self._polytope = polytope
        self._control_plane = control_plane
        self._inline_serving = inline_serving

    @property
    def verification(self) -> VerificationService:
        if self._verification is not None:
            return self._verification
        if self._control_plane is None:
            raise RuntimeClosedError(
                "inline serving runtime has no verification service"
            )
        return self._control_plane.verification

    @property
    def polytope(self) -> PolytopeService:
        if self._polytope is not None:
            return self._polytope
        if self._control_plane is None:
            raise RuntimeClosedError("inline serving runtime has no polytope service")
        return self._control_plane.polytope

    @property
    def operations(self) -> CatalogOperationCollector | OperationDispatcher:
        bound = self.core.operations
        if bound is None:
            raise RuntimeClosedError("runtime operations have not been assigned")
        return bound

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


__all__ = ["InlineServingResources", "JacobianRuntime", "RuntimeClosedError"]
