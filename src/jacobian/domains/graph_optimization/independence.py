"""Thin operation binding for bounded independence-number search."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiagnostic
from jacobian.domains._examples import example
from jacobian.math.graphs.independence import (
    IndependenceNumberRequest,
    IndependenceNumberResult,
    independence_number,
)
from jacobian.operation_bindings import inline_operation
from jacobian.operations import OperationRefusalError, OperationSpec

_INVALID_REQUEST = CapabilityDiagnostic(
    code="INVALID_GRAPH_INDEPENDENCE_NUMBER_REQUEST",
    stage="graph_independence_number_input_validation",
    message=(
        "Input does not satisfy the bounded finite simple-graph independence contract."
    ),
    hint="Supply a canonical simple graph of order at most 128.",
)


def _execute(request: IndependenceNumberRequest) -> IndependenceNumberResult:
    try:
        return independence_number(request)
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        raise OperationRefusalError(
            CapabilityDiagnostic(
                code="GRAPH_INDEPENDENCE_NUMBER_SEARCH_FAILED",
                stage="graph_independence_number_computation",
                message=str(exc),
                hint="Reduce the graph or increase the declared wall-clock budget.",
            )
        ) from exc


INDEPENDENCE_NUMBER_CAPABILITY = inline_operation(
    OperationSpec(
        operation_id="graph.invariant.independence_number.compute",
        version="2",
        title="Independence number",
        description=(
            "Compute a maximum edge-free vertex subset through order 128. Return "
            "either the exact optimum or a feasible incumbent with explicit lower "
            "and upper bounds when the wall-clock budget expires."
        ),
        request_type=IndependenceNumberRequest,
        result_type=IndependenceNumberResult,
        execute=_execute,
        tags=(
            "graph",
            "invariant",
            "independent-set",
            "independence-number",
            "maximum",
            "bounded",
            "z3",
        ),
        invalid_request=_INVALID_REQUEST,
        invocation_examples=(
            example(
                "cycle_five",
                "Compute the independence number of a five-cycle.",
                {
                    "graph": {
                        "vertices": ["0", "1", "2", "3", "4"],
                        "edges": [
                            ["0", "1"],
                            ["0", "4"],
                            ["1", "2"],
                            ["2", "3"],
                            ["3", "4"],
                        ],
                    },
                    "resource_budget": {"wall_seconds": 5, "max_order": 128},
                },
            ),
        ),
    )
)

__all__ = ["INDEPENDENCE_NUMBER_CAPABILITY"]
