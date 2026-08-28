"""Graphical model operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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
    return DSeparationResult._from_kernel(
        request,
        d_separation(
            request.variable_count,
            request.edges,
            request.set_a,
            request.set_b,
            request.set_c,
        ),
    )


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
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
    _op(
        "graphical_model.factor.multiply",
        "Multiply two factors",
        "Compute the product of two factors over the union of their variables "
        "using bounded exact rational arithmetic. Scalar factors use an empty scope.",
        FactorMultiplyRequest,
        FactorMultiplyResult,
        _factor_multiply,
        "graphical-model",
        "factor",
        "exact",
        examples=(
            example(
                "multiply_two_factors",
                "Multiply a 2-var factor by another 2-var factor.",
                {
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
    _op(
        "graphical_model.factor.marginalize",
        "Marginalize out a variable from a factor",
        "Sum out a variable from a factor, producing a factor over the "
        "remaining variables using exact rational arithmetic.",
        FactorMarginalizeRequest,
        FactorMarginalizeResult,
        _factor_marginalize,
        "graphical-model",
        "factor",
        "marginalization",
        "exact",
        examples=(
            example(
                "marginalize_var_0",
                "Marginalize out variable 0 from a single-variable factor.",
                {
                    "factor": _FACTOR_SINGLE,
                    "variable": 0,
                },
            ),
        ),
    ),
    _op(
        "graphical_model.d_separation.compute",
        "Check d-separation in a Bayesian network",
        "Check whether two sets of variables are d-separated given a "
        "conditioning set in a bounded directed acyclic graph, using ancestral "
        "restriction and moralization.",
        DSeparationRequest,
        DSeparationResult,
        _d_separation,
        "graphical-model",
        "d-separation",
        "exact",
        examples=(
            example(
                "conditioned_chain",
                "Decide whether endpoints of a three-node chain are separated by its middle node.",
                {
                    "variable_count": 3,
                    "edges": [[0, 1], [1, 2]],
                    "set_a": [0],
                    "set_b": [2],
                    "set_c": [1],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
