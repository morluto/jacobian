"""Bounded complete colored-family declaration."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.enumeration._models import (
    ConnectedColoredGraphFamily,
    ConnectedColoredGraphsRequest,
)
from jacobian.math.graphs.enumeration.operations import connected_colored_graphs

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.colored.connected_family.enumerate",
        title="Enumerate connected colored graphs under edge-type costs",
        description="Return one representative of every color-preserving isomorphism class of connected simple graphs with 2 through vertex_bound vertices and total edge cost at most cost_bound. Singletons are excluded; unused palette colors are allowed. Missing unordered color pairs forbid edges. Uses the complete atlas through seven vertices and exact coloring orbits; at most 20000 candidate colorings and 150000000 search units. Colors retain their exact names; output is never truncated.",
        request_type=ConnectedColoredGraphsRequest,
        result_type=ConnectedColoredGraphFamily,
        run=connected_colored_graphs,
        tags=(
            "graph",
            "enumeration",
            "colored",
            "connected",
            "isomorphism",
            "edge-cost",
        ),
        examples=(
            OperationExample(
                name="two_color_edge_family",
                description="Enumerate connected colored graphs of cost at most two; allowed unordered color pairs are sorted and carry positive costs.",
                input={
                    "palette": ["outside", "root"],
                    "edge_costs": [
                        {"colors": ["outside", "outside"], "cost": 2},
                        {"colors": ["outside", "root"], "cost": 1},
                    ],
                    "vertex_bound": 3,
                    "cost_bound": 2,
                },
            ),
        ),
    ),
)
