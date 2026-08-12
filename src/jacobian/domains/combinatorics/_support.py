"""Combinatorics operation declarations."""

from jacobian.operation_bindings import DurableOperationFactory, InlineOperationFactory
from jacobian.operations import OperationFailure

_FAILURE = OperationFailure(
    code="COMBINATORICS_OPERATION_NOT_APPLICABLE",
    stage="combinatorics_computation",
    hint="Check the bounded combinatorics input and mathematical preconditions.",
    exceptions=(TypeError, ValueError, ArithmeticError),
)

combinatorics_operation = InlineOperationFactory(_FAILURE)
materialized_combinatorics_operation = DurableOperationFactory(_FAILURE)
