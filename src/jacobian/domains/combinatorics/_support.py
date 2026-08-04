"""Combinatorics operation declarations."""

from jacobian.operations import (
    ComputedOperationFactory,
    MaterializedOperationFactory,
    OperationFailure,
)

_FAILURE = OperationFailure(
    code="COMBINATORICS_OPERATION_NOT_APPLICABLE",
    stage="combinatorics_computation",
    hint="Check the bounded combinatorics input and mathematical preconditions.",
    exceptions=(TypeError, ValueError, ArithmeticError),
)

combinatorics_operation = ComputedOperationFactory(_FAILURE)
materialized_combinatorics_operation = MaterializedOperationFactory(_FAILURE)
