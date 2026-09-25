"""Exact complete assignment evaluation for named finite CSP constraints."""

import json

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.relational_structures import (
    CspAssignmentRequest,
    FiniteCspInstance,
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    profile_csp_assignment,
)


def _instance() -> FiniteCspInstance:
    return FiniteCspInstance(
        template=FiniteRelationalStructure(
            carrier_size=2,
            signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
            relation_tables=(((0, 1),),),
        ),
        variable_count=2,
        constraints=(
            {"constraint_id": "first", "symbol_id": "E", "scope": (0, 1)},
            {"constraint_id": "duplicate", "symbol_id": "E", "scope": (0, 1)},
            {"constraint_id": "reverse", "symbol_id": "E", "scope": (1, 0)},
        ),
    )


def test_assignment_profile_preserves_occurrences_and_first_failure() -> None:
    result = profile_csp_assignment(_instance(), (0, 1))

    assert result.status == "NOT_A_SOLUTION"
    assert [item.constraint_id for item in result.evaluations] == [
        "first",
        "duplicate",
        "reverse",
    ]
    assert [item.allowed for item in result.evaluations] == [True, True, False]
    assert result.evaluations[-1].target_tuple == (1, 0)
    assert result.first_violation == result.evaluations[-1]


def test_valid_assignment_is_exact_solution_and_empty_instance_is_solved() -> None:
    instance = FiniteCspInstance(
        template=_instance().template,
        variable_count=2,
        constraints=({"constraint_id": "edge", "symbol_id": "E", "scope": (0, 1)},),
    )
    result = profile_csp_assignment(instance, (0, 1))
    assert result.status == "SOLUTION"
    assert result.first_violation is None

    empty = FiniteCspInstance(
        template=FiniteRelationalStructure(carrier_size=0),
        variable_count=0,
        constraints=(),
    )
    empty_result = profile_csp_assignment(empty, ())
    assert empty_result.status == "SOLUTION"
    assert empty_result.evaluations == ()


def test_catalog_exposes_direct_assignment_profile() -> None:
    operation_id = "csp.assignment.profile.compute"
    assert any(tool.operation_id == operation_id for tool in BUILTIN_TOOLS)
    payload = CspAssignmentRequest(
        instance=_instance(), assignment=(0, 1)
    ).model_dump_json()
    output = invoke_operation(operation_id, json.loads(payload), Catalog.open()).output
    assert output["status"] == "NOT_A_SOLUTION"
    assert output["first_violation"]["constraint_id"] == "reverse"


def test_native_rejects_malformed_assignments_without_a_wire_request() -> None:
    with pytest.raises(OperationDomainValidationError) as axis:
        profile_csp_assignment(_instance(), (0, 1, 0))
    assert axis.value.errors()[0]["type"] == "relational.csp.assignment_shape"
    with pytest.raises(OperationDomainValidationError) as value:
        profile_csp_assignment(_instance(), (0, 2))
    assert value.value.errors()[0]["type"] == "relational.csp.assignment_shape"
    with pytest.raises(OperationDomainValidationError) as shape:
        profile_csp_assignment(_instance(), "ab")
    assert shape.value.errors()[0]["type"] == "relational.csp.assignment_shape"
    with pytest.raises(OperationDomainValidationError) as wrong_type:
        profile_csp_assignment(_instance(), (0, "a"))
    assert wrong_type.value.errors()[0]["type"] == "relational.csp.assignment_shape"
