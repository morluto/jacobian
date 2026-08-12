"""Executable contracts for examples advertised by the core domain catalog."""

from __future__ import annotations

import pytest

from jacobian.domains.arithmetic import build_arithmetic_bundle
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.finite_sets import build_finite_set_bundle
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.domains.sequences import build_sequence_bundle

_CORE_OPERATIONS = tuple(
    operation
    for bundle in (
        build_arithmetic_bundle(),
        build_combinatorics_bundle(),
        build_finite_set_bundle(),
        build_number_theory_bundle(),
        build_sequence_bundle(),
    )
    for operation in bundle.capabilities
)


@pytest.mark.parametrize(
    "operation",
    _CORE_OPERATIONS,
    ids=lambda operation: operation.spec.operation_id,
)
def test_advertised_invocation_example_executes_successfully(operation) -> None:
    operation_id = operation.spec.operation_id
    examples = operation.spec.invocation_examples
    assert examples, f"{operation_id} must advertise one executable example"
    request = operation.spec.request_type.model_validate(examples[0].input)
    outcome = operation.spec.execute(request)
    assert isinstance(outcome, operation.spec.result_type), (operation_id, outcome)
