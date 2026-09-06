"""Graphical model operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.probability.graphical_models._models import (
    DSeparationRequest,
    DSeparationResult,
    FactorMarginalizeRequest,
    FactorMarginalizeResult,
    FactorMultiplyRequest,
    FactorMultiplyResult,
)
from jacobian.math.probability.graphical_models.operations import (
    d_separation,
    factor_marginalize,
    factor_multiply,
)


def _factor_multiply(request: FactorMultiplyRequest) -> FactorMultiplyResult:
    return FactorMultiplyResult._from_kernel(
        request.left, request.right, factor_multiply(request.left, request.right)
    )


def _factor_marginalize(
    request: FactorMarginalizeRequest,
) -> FactorMarginalizeResult:
    return FactorMarginalizeResult._from_kernel(
        request.factor,
        request.variable,
        factor_marginalize(request.factor, request.variable),
    )


def _d_separation(request: DSeparationRequest) -> DSeparationResult:
    query = request.query
    return DSeparationResult._from_kernel(
        request,
        d_separation(
            query.dag.variable_count,
            query.dag.edges,
            query.set_a,
            query.set_b,
            query.set_c,
        ),
    )


_FACTOR1 = {
    "variables": [0, 1],
    "domain_sizes": [2, 2, 2],
    "table": [
        {"num": "1", "den": "2"},
        {"num": "1", "den": "2"},
        {"num": "1", "den": "3"},
        {"num": "2", "den": "3"},
    ],
}

_FACTOR_SINGLE = {
    "variables": [0],
    "domain_sizes": [2],
    "table": [{"num": "1", "den": "3"}, {"num": "2", "den": "3"}],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graphical_model.factor.multiply",
        title="Multiply two factors",
        description="Compute the product of two factors over the union of their variables "
        "using bounded exact rational arithmetic. Scalar factors use an empty scope.",
        request_type=FactorMultiplyRequest,
        result_type=FactorMultiplyResult,
        run=_factor_multiply,
        tags=("graphical-model", "factor", "exact"),
        examples=(
            OperationExample(
                name="multiply_two_factors",
                description="Multiply a 2-var factor by another 2-var factor.",
                input={
                    "left": _FACTOR1,
                    "right": {
                        "variables": [1],
                        "domain_sizes": [2, 2, 2],
                        "table": [
                            {"num": "1", "den": "2"},
                            {"num": "1", "den": "2"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graphical_model.factor.marginalize",
        title="Marginalize out a variable from a factor",
        description="Sum out a variable from a factor, producing a factor over the "
        "remaining variables using exact rational arithmetic.",
        request_type=FactorMarginalizeRequest,
        result_type=FactorMarginalizeResult,
        run=_factor_marginalize,
        tags=("graphical-model", "factor", "marginalization", "exact"),
        examples=(
            OperationExample(
                name="marginalize_var_0",
                description="Marginalize out variable 0 from a single-variable factor.",
                input={
                    "factor": _FACTOR_SINGLE,
                    "variable": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graphical_model.d_separation.compute",
        title="Check d-separation in a Bayesian network",
        description="Check whether two sets of variables are d-separated given a "
        "conditioning set in a bounded directed acyclic graph, using ancestral "
        "restriction and moralization.",
        request_type=DSeparationRequest,
        result_type=DSeparationResult,
        run=_d_separation,
        tags=("graphical-model", "d-separation", "exact"),
        examples=(
            OperationExample(
                name="conditioned_chain",
                description="Decide whether endpoints of a three-node chain are separated by its middle node.",
                input={
                    "query": {
                        "dag": {"variables": [0, 1, 2], "edges": [[0, 1], [1, 2]]},
                        "set_a": [0],
                        "set_b": [2],
                        "set_c": [1],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
