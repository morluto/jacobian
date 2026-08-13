"""Sequence operation declarations."""

from jacobian.operation_declarations import (
    InlineOperationFactory,
    OperationFailure,
)

sequence_operation = InlineOperationFactory(
    OperationFailure(
        code="SEQUENCE_OPERATION_NOT_APPLICABLE",
        stage="sequence_computation",
        hint="Check the operation's sequence preconditions.",
        exceptions=(TypeError, ValueError, ArithmeticError),
    )
)
