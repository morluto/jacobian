"""Domain adapter for graphical model operations."""

from __future__ import annotations

from jacobian.math.graphical_models._models import (
    DSeparationRequest,
    DSeparationResult,
    FactorMarginalizeRequest,
    FactorMarginalizeResult,
    FactorMultiplyRequest,
    FactorMultiplyResult,
    VariableEliminationRequest,
    VariableEliminationResult,
)
from jacobian.math.graphical_models.operations import (
    d_separation,
    factor_marginalize,
    factor_multiply,
    variable_elimination,
)

__all__ = [
    "compute_d_separation",
    "compute_factor_marginalize",
    "compute_factor_multiply",
    "compute_variable_elimination",
]


def compute_factor_multiply(request: FactorMultiplyRequest) -> FactorMultiplyResult:
    return FactorMultiplyResult(
        left=request.left,
        right=request.right,
        factor=factor_multiply(request.left, request.right),
    )


def compute_factor_marginalize(
    request: FactorMarginalizeRequest,
) -> FactorMarginalizeResult:
    return FactorMarginalizeResult(
        source_factor=request.factor,
        variable=request.variable,
        factor=factor_marginalize(request.factor, request.variable),
    )


def compute_variable_elimination(
    request: VariableEliminationRequest,
) -> VariableEliminationResult:
    result = variable_elimination(
        list(request.factors),
        request.domain_sizes,
        request.elimination_order,
        request.query_variables,
    )
    return VariableEliminationResult(
        factors=request.factors,
        domain_sizes=request.domain_sizes,
        elimination_order=request.elimination_order,
        query_variables=request.query_variables,
        factor=result,
    )


def compute_d_separation(request: DSeparationRequest) -> DSeparationResult:
    return DSeparationResult(
        variable_count=request.variable_count,
        edges=request.edges,
        set_a=request.set_a,
        set_b=request.set_b,
        set_c=request.set_c,
        d_separated=d_separation(
            request.variable_count,
            request.edges,
            request.set_a,
            request.set_b,
            request.set_c,
        ),
    )
