"""Public declaration for exact lattice-gauge path holonomy."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.gauge._models import HolonomyRequest, HolonomyResult
from jacobian.math.gauge.operations import path_holonomy


def _run_holonomy(request: HolonomyRequest) -> HolonomyResult:
    return path_holonomy(request.field, request.path)


TOOLS = (
    MathTool(
        operation_id="lattice_gauge.holonomy.compute",
        title="Compute the ordered exact holonomy of a lattice gauge field",
        description=(
            "For a finite oriented gauge lattice with permutation-valued edge "
            "labels and an oriented edge path chaining head-to-tail, return "
            "the ordered exact group product Hol = U_{e_1} ... U_{e_m} under "
            "the left-to-right convention with the per-edge contribution map "
            "and start/end vertices. Backward steps resolve to exact inverses, "
            "so reverse traversal gives the inverse holonomy."
        ),
        request_type=HolonomyRequest,
        result_type=HolonomyResult,
        run=_run_holonomy,
        tags=("lattice-gauge", "holonomy", "exact"),
        discovery_terms=(
            "lattice gauge path holonomy",
            "Wilson line ordered product",
            "gauge field edge product",
        ),
        examples=(
            OperationExample(
                name="triangle_plaquette_holonomy",
                description=(
                    "Compute the exact S3 holonomy around an oriented triangle; "
                    "path steps must chain head-to-tail over labelled edges."
                ),
                input={
                    "field": {
                        "lattice": {
                            "vertices": ["a", "b", "c"],
                            "edges": [
                                {"edge_id": "ab", "tail": "a", "head": "b"},
                                {"edge_id": "bc", "tail": "b", "head": "c"},
                                {"edge_id": "ca", "tail": "c", "head": "a"},
                            ],
                        },
                        "degree": 3,
                        "edge_labels": [
                            {
                                "edge_id": "ab",
                                "label": {"degree": 3, "image": [1, 2, 0]},
                            },
                            {
                                "edge_id": "bc",
                                "label": {"degree": 3, "image": [1, 2, 0]},
                            },
                            {
                                "edge_id": "ca",
                                "label": {"degree": 3, "image": [1, 2, 0]},
                            },
                        ],
                    },
                    "path": {
                        "steps": [
                            {"edge_id": "ab", "forward": True},
                            {"edge_id": "bc", "forward": True},
                            {"edge_id": "ca", "forward": True},
                        ]
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
