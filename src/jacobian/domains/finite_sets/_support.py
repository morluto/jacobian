"""Finite-set operation declarations."""

from jacobian.operation_declarations import (
    InlineOperationFactory,
    OperationFailure,
)

finite_set_operation = InlineOperationFactory(
    OperationFailure(
        code="FINITE_SET_OPERATION_NOT_APPLICABLE",
        stage="finite_set_computation",
        hint="Check the operation's finite-set preconditions.",
    )
)
