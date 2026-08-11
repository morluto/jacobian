"""Executable contracts for examples advertised by the core domain catalog."""

from __future__ import annotations

import pytest

from jacobian.domains.arithmetic import build_arithmetic_bundle
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.finite_sets import build_finite_set_bundle
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.domains.sequences import build_sequence_bundle
from jacobian.operations import BoundedSearchWitness, ComputedSuccess

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
    ids=lambda operation: operation.capability_id,
)
def test_advertised_invocation_example_executes_successfully(operation) -> None:
    assert operation.invocation_examples, (
        f"{operation.capability_id} must advertise one executable example"
    )
    example = operation.invocation_examples[0]
    request = operation.request_model.model_validate(example.input)

    outcome = operation.implementation(request)

    assert isinstance(outcome, ComputedSuccess | BoundedSearchWitness), (
        operation.capability_id,
        outcome,
    )
    assert isinstance(outcome.value, operation.result_model)
