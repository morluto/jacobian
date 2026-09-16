"""Public declaration for exact link-diagram component traversal."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.links._models import (
    LinkComponentsRequest,
    LinkComponentsResult,
)
from jacobian.math.topology.links.operations import link_components


def _run_link_components(request: LinkComponentsRequest) -> LinkComponentsResult:
    return link_components(request.diagram)


TOOLS = (
    MathTool(
        operation_id="link_diagram.components.compute",
        title="Partition a classical oriented link diagram into components",
        description=(
            "For a well-formed classical oriented link diagram with "
            "dart/half-edge crossings carrying over/under pairs and an arc "
            "involution, return the complete oriented component partition: "
            "cyclic dart sequences with per-crossing OVER/UNDER roles and "
            "component lengths. Every half-edge lies in exactly one crossing "
            "and one arc; traversal gives disjoint cycles covering every arc "
            "exactly once."
        ),
        request_type=LinkComponentsRequest,
        result_type=LinkComponentsResult,
        run=_run_link_components,
        tags=("link-diagram", "components", "exact"),
        discovery_terms=(
            "link diagram components",
            "knot diagram traversal",
            "crossing over under roles",
        ),
        examples=(
            OperationExample(
                name="hopf_link_two_components",
                description=(
                    "Partition the two-crossing Hopf link into two components; "
                    "every half-edge must lie in exactly one crossing and one "
                    "arc."
                ),
                input={
                    "diagram": {
                        "crossings": [
                            {
                                "crossing_id": "c0",
                                "half_edges": ["a0", "b0", "a1", "b1"],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                            },
                            {
                                "crossing_id": "c1",
                                "half_edges": ["a2", "b2", "a3", "b3"],
                                "over_pair": [0, 2],
                                "under_pair": [1, 3],
                            },
                        ],
                        "arcs": [
                            {"first": "a1", "second": "b2"},
                            {"first": "a3", "second": "b0"},
                            {"first": "b1", "second": "a2"},
                            {"first": "b3", "second": "a0"},
                        ],
                        "free_loops": 0,
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
