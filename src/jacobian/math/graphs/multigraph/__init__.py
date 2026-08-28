"""Native finite-multigraph operations."""

from jacobian.math.graphs.multigraph.operations import (
    cycle_multicover,
    eulerian_cycles,
    multigraph_flow_check,
    multigraph_flow_find,
)

__all__ = [
    "cycle_multicover",
    "eulerian_cycles",
    "multigraph_flow_check",
    "multigraph_flow_find",
]
