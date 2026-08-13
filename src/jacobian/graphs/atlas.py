"""Process-local immutable access to NetworkX Graph Atlas representatives.

The NetworkX backend is loaded lazily through :data:`networkx_loader` so that
importing this module (and the graph operation modules that depend on it) does
not import NetworkX. The backend is loaded on first invocation of
:func:`graph_atlas_order`.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

from jacobian.providers import LazyLoader

if TYPE_CHECKING:
    import networkx as nx

_MAX_ATLAS_ORDER = 7


def _load_networkx() -> Any:
    """Import and return the NetworkX module on first use."""

    import networkx as nx

    return nx


#: Single lazy owner of the NetworkX backend shared by the graph operation
#: modules. Importing this module does not import NetworkX; the backend is
#: loaded on the first :meth:`LazyLoader.get` call.
networkx_loader: LazyLoader[Any] = LazyLoader(_load_networkx, component_id="networkx")


@cache
def _graph_atlas_by_order() -> tuple[tuple[nx.Graph[Any], ...], ...]:
    backend = networkx_loader.get()
    grouped: list[list[nx.Graph[Any]]] = [[] for _ in range(_MAX_ATLAS_ORDER + 1)]
    for graph in backend.graph_atlas_g():
        grouped[graph.number_of_nodes()].append(backend.freeze(graph))
    return tuple(tuple(graphs) for graphs in grouped)


def graph_atlas_order(order: int) -> tuple[nx.Graph[Any], ...]:
    """Return frozen atlas representatives of one supported order."""

    if not 0 <= order <= _MAX_ATLAS_ORDER:
        raise ValueError("Graph Atlas order must be between zero and seven")
    return _graph_atlas_by_order()[order]
