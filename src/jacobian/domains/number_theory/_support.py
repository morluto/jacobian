"""Number-theory operation declarations."""

from jacobian.operation_declarations import (
    InlineOperationFactory,
    OperationFailure,
)

_NUMBER_THEORY_FAILURE = OperationFailure(
    code="NUMBER_THEORY_OPERATION_NOT_APPLICABLE",
    stage="number_theory_computation",
    hint="Check divisibility, positivity, primality, and modular preconditions.",
)

number_theory_operation = InlineOperationFactory(_NUMBER_THEORY_FAILURE)
