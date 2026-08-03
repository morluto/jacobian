"""Number-theory operation declarations."""

from jacobian.operations import (
    ComputedOperationFactory,
    MaterializedOperationFactory,
    OperationFailure,
)

_NUMBER_THEORY_FAILURE = OperationFailure(
    code="NUMBER_THEORY_OPERATION_NOT_APPLICABLE",
    stage="number_theory_computation",
    hint="Check divisibility, positivity, primality, and modular preconditions.",
)

number_theory_operation = ComputedOperationFactory(_NUMBER_THEORY_FAILURE)
materialized_number_theory_operation = MaterializedOperationFactory(
    _NUMBER_THEORY_FAILURE
)
