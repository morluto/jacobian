"""Deterministic H-minor-model checker declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.minors._models import (
    MinorModelCheckRequest,
    MinorModelCheckResult,
)
from jacobian.math.graphs.minors.operations import check_minor_model


def _check(request: MinorModelCheckRequest) -> MinorModelCheckResult:
    return check_minor_model(
        request.source, request.target, request.branch_sets, request.witnesses
    )


TOOLS = (
    MathTool(
        operation_id="graph.minor_model.check",
        title="Check an explicit H-minor model in G",
        description=(
            "Deterministically check one candidate H-minor model: every target "
            "vertex needs exactly one nonempty branch set, branch sets are "
            "pairwise disjoint, each induces a connected source subgraph, and "
            "every target edge needs a source-edge witness crossing its two "
            "branch sets. VALID_MINOR_MODEL returns canonical branch sets, "
            "per-branch connectivity ledgers, the witness ledger, and used "
            "versus deleted source structure; INVALID_MINOR_MODEL returns the "
            "first concrete obstruction. Extra source edges between branch "
            "sets stay legal for ordinary minors."
        ),
        request_type=MinorModelCheckRequest,
        result_type=MinorModelCheckResult,
        run=_check,
        tags=("graph", "minor", "branch-set", "checker", "exact"),
        discovery_terms=("minor model", "branch sets", "graph minor"),
        examples=(
            OperationExample(
                name="edge_in_a_triangle",
                description="A single target edge modelled by two singleton branch sets in a triangle.",
                input={
                    "source": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                    },
                    "target": {"vertices": ["x", "y"], "edges": [["x", "y"]]},
                    "branch_sets": [
                        {"target": "x", "members": ["a"]},
                        {"target": "y", "members": ["b"]},
                    ],
                    "witnesses": [{"targets": ["x", "y"], "source_edge": ["a", "b"]}],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
