"""Deletion-partition compatibility closure operation declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.coloring.deletion_partition_closure._models import (
    DeletionPartitionClosure,
    DeletionPartitionClosureRequest,
)
from jacobian.math.graphs.coloring.deletion_partition_closure.operations import (
    construct,
)


def compute(request: DeletionPartitionClosureRequest) -> DeletionPartitionClosure:
    return construct(request.template)


TOOLS = (
    MathTool(
        operation_id="graph.coloring.deletion_partition_closure.construct",
        title="Construct a vertex-deletion partition compatibility closure",
        description=(
            "Construct the largest simple graph for which every supplied partition "
            "of V minus v is a proper coloring after deleting v. A pair is an edge "
            "exactly when every applicable row separates its endpoints. Return the "
            "canonical source template, complete pair axis and first-row/block "
            "blocker for every nonedge. Accept up to 256 ordered labels of 64 UTF-8 "
            "bytes, including empty carriers and vacuous two-vertex closures. "
            "This does not decide chromatic number or vertex criticality."
        ),
        request_type=DeletionPartitionClosureRequest,
        result_type=DeletionPartitionClosure,
        run=compute,
        tags=("graph", "coloring", "partition", "compatibility", "closure"),
        examples=(
            OperationExample(
                name="one_block_deletion_rows",
                description="Every pair in a three-vertex carrier has a blocking row.",
                input={
                    "template": {
                        "vertices": ["a", "b", "c"],
                        "block_bound": "1",
                        "rows": [
                            {"deleted_vertex": "a", "blocks": [["b", "c"]]},
                            {"deleted_vertex": "b", "blocks": [["a", "c"]]},
                            {"deleted_vertex": "c", "blocks": [["a", "b"]]},
                        ],
                    }
                },
            ),
        ),
    ),
)
