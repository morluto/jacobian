"""Executable contracts for examples advertised by the core domain catalog."""

from __future__ import annotations

import pytest

from jacobian.domains.arithmetic import arithmetic_operations
from jacobian.domains.combinatorics import combinatorics_operations
from jacobian.domains.finite_sets import finite_set_operations
from jacobian.domains.number_theory import number_theory_operations
from jacobian.domains.sequences import sequence_operations

_CORE_OPERATIONS = tuple(
    operation
    for declarations in (
        arithmetic_operations(),
        combinatorics_operations(),
        finite_set_operations(),
        number_theory_operations(),
        sequence_operations(),
    )
    for operation in declarations
)


@pytest.mark.parametrize(
    "operation",
    _CORE_OPERATIONS,
    ids=lambda operation: operation.operation_id,
)
def test_advertised_invocation_example_executes_successfully(operation) -> None:
    operation_id = operation.operation_id
    examples = operation.examples
    assert examples, f"{operation_id} must advertise one executable example"
    request = operation.request_type.model_validate(examples[0].input)
    outcome = operation.run(request)
    assert isinstance(outcome, operation.result_type), (operation_id, outcome)
