"""Directed-graph operation ownership.

The operation implementations use NetworkX, so keep them lazy here.  Values
owned by other mathematical domains may reuse the directed graph contracts
without importing a packaged backend merely by importing this package.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jacobian.math.graphs.directed.operations import (
        acyclic_order,
        condensation,
        dag_longest_path,
        reachability,
        strongly_connected_components,
        verify_acyclic_order,
        verify_condensation,
        verify_dag_longest_path,
        verify_reachability,
        verify_strongly_connected_components,
    )

__all__ = [
    "acyclic_order",
    "condensation",
    "dag_longest_path",
    "reachability",
    "strongly_connected_components",
    "verify_acyclic_order",
    "verify_condensation",
    "verify_dag_longest_path",
    "verify_reachability",
    "verify_strongly_connected_components",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module("jacobian.math.graphs.directed.operations"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
