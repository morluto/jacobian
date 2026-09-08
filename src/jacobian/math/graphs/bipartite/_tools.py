"""Publication of the source-bound Dulmage--Mendelsohn decomposition."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.bipartite._models import (
    DulmageMendelsohnDecomposition,
    DulmageMendelsohnRequest,
)
from jacobian.math.graphs.bipartite.operations import dulmage_mendelsohn


def _run(request: DulmageMendelsohnRequest) -> DulmageMendelsohnDecomposition:
    return dulmage_mendelsohn(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.bipartite.dulmage_mendelsohn.compute",
        title="Dulmage-Mendelsohn decomposition with fixed bipartite sides",
        description=(
            "Return left-excess and right-excess deficient regions, balanced "
            "elementary blocks, structural rank and the direct condensation DAG "
            "of a finite bipartite graph with explicit ordered sides. Retains "
            "original indexed vertex identities, isolated vertices and empty "
            "sides. Orient all edges left-to-right and matching edges also "
            "right-to-left. Balanced SCC blocks are labelled by first input-left "
            "position; inter-block arcs follow this orientation, without "
            "transitive closure. Structural results are independent of the "
            "chosen maximum matching. Matching/traversal/output work is admitted "
            "before execution under one 30-second safety deadline."
        ),
        request_type=DulmageMendelsohnRequest,
        result_type=DulmageMendelsohnDecomposition,
        run=_run,
        tags=(
            "graph",
            "bipartite",
            "Dulmage-Mendelsohn",
            "matching",
            "structural rank",
            "decomposition",
            "exact",
        ),
        examples=(
            OperationExample(
                name="triangular_pattern",
                description="Compute the Dulmage-Mendelsohn decomposition of the 2-by-2 triangular pattern, yielding two balanced blocks and the arc 0→1; the declared sides must partition every vertex and every edge must cross those sides.",
                input={
                    "graph": {
                        "graph": {"vertex_count": 4, "edges": [[0, 2], [0, 3], [1, 3]]},
                        "left_vertices": [0, 1],
                        "right_vertices": [2, 3],
                    }
                },
            ),
        ),
    ),
)
