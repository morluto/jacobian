"""Construct the immutable packaged inline operation runtime."""

from __future__ import annotations

from jacobian.operation_dispatcher import OperationDispatcher
from jacobian.package_index import PackageIndexRegistry
from jacobian.runtime.model import InlineServingResources, JacobianRuntime
from jacobian.serving_catalog import ServingCatalog


def create_inline_serving_runtime(catalog: ServingCatalog) -> JacobianRuntime:
    """Serve packaged direct operations without workspace state."""

    dispatcher = OperationDispatcher(catalog, PackageIndexRegistry(catalog.index))
    return JacobianRuntime(InlineServingResources(dispatcher))


__all__ = ["create_inline_serving_runtime"]
