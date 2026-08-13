"""Geometry operation declarations."""

from jacobian.operation_declarations import (
    InlineOperationFactory,
    OperationFailure,
)

geometry_operation = InlineOperationFactory(
    OperationFailure(
        code="GEOMETRY_OPERATION_NOT_APPLICABLE",
        stage="geometry_computation",
        hint="Check the operation's nondegeneracy preconditions.",
    )
)
