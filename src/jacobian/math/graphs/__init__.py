"""Supported exact finite simple-graph API."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math.graphs.independence import (
        IndependenceNumberResult,
        independence_number,
    )
    from jacobian.math.graphs.operations import (
        biconnected_components,
        compose_graphs,
        diameter,
        explicit_graph,
        is_eulerian,
        radius,
        strongly_connected_components,
        triangle_count,
    )
    from jacobian.math.graphs.values import (
        ColoredUndirectedGraph,
        IndexedSimpleUndirectedGraph,
        SimpleUndirectedGraph,
    )

__all__ = [
    "ColoredUndirectedGraph",
    "IndependenceNumberResult",
    "IndexedSimpleUndirectedGraph",
    "SimpleUndirectedGraph",
    "biconnected_components",
    "compose_graphs",
    "diameter",
    "explicit_graph",
    "independence_number",
    "is_eulerian",
    "radius",
    "strongly_connected_components",
    "triangle_count",
]

_OWNER_MODULES = {
    "ColoredUndirectedGraph": "values",
    "IndexedSimpleUndirectedGraph": "values",
    "SimpleUndirectedGraph": "values",
    "IndependenceNumberResult": "independence",
    "independence_number": "independence",
}


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    owner = _OWNER_MODULES.get(name, "operations")
    value = getattr(import_module(f"{__name__}.{owner}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
