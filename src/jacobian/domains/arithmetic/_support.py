"""Arithmetic operation declarations."""

from jacobian.operation_bindings import InlineOperationFactory
from jacobian.operations import OperationFailure

arithmetic_operation = InlineOperationFactory(
    OperationFailure(
        code="ARITHMETIC_OPERATION_NOT_APPLICABLE",
        stage="arithmetic_computation",
        hint="Check the operation's exact-arithmetic preconditions.",
    )
)
