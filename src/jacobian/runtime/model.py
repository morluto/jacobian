"""Lifecycle for the immutable inline operation runtime."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.operation_dispatcher import OperationDispatcher


@dataclass(slots=True)
class InlineServingResources:
    """The dispatcher is the only resource owned by stateless serving."""

    operations: OperationDispatcher

    def close(self) -> None:
        self.operations.close()


class RuntimeClosedError(RuntimeError):
    """An operation requires a live Jacobian runtime."""


class JacobianRuntime:
    """Own one immutable packaged operation dispatcher."""

    def __init__(self, core: InlineServingResources) -> None:
        self._closed = False
        self.core = core

    @property
    def operations(self) -> OperationDispatcher:
        if self._closed:
            raise RuntimeClosedError("runtime is closed")
        return self.core.operations

    def close(self) -> None:
        if not self._closed:
            self.core.close()
            self._closed = True

    def __enter__(self) -> JacobianRuntime:
        if self._closed:
            raise RuntimeClosedError("runtime is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["InlineServingResources", "JacobianRuntime", "RuntimeClosedError"]
