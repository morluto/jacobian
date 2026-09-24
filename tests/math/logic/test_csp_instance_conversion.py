"""Exact CSP-to-relational-structure conversion contracts."""

from __future__ import annotations

import json
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.relational_structures import (
    FiniteCspInstance,
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    count_homomorphisms,
    csp_instance_to_source_structure,
)


def _instance() -> FiniteCspInstance:
    return FiniteCspInstance(
        template=FiniteRelationalStructure(
            carrier_size=2,
            signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
            relation_tables=(((0, 0), (0, 1), (1, 0)),),
        ),
        variable_count=3,
        constraints=(
            {"constraint_id": "c0", "symbol_id": "E", "scope": (0, 1)},
            {"constraint_id": "c1", "symbol_id": "E", "scope": (1, 0)},
            {"constraint_id": "c2", "symbol_id": "E", "scope": (2, 2)},
            # A second named occurrence is retained but deduplicates as a
            # relation row, as required by canonical-database semantics.
            {"constraint_id": "c3", "symbol_id": "E", "scope": (0, 1)},
        ),
    )


def test_conversion_preserves_repeated_variables_and_constraint_provenance() -> None:
    instance = _instance()
    source = csp_instance_to_source_structure(instance)
    assert source.carrier_size == 3
    assert source.signature == instance.template.signature
    assert source.relation_tables == (((0, 1), (1, 0), (2, 2)),)
    assert [constraint.constraint_id for constraint in instance.constraints] == [
        "c0",
        "c1",
        "c2",
        "c3",
    ]


def test_homomorphisms_are_exactly_directly_enumerated_csp_solutions() -> None:
    instance = _instance()
    source = csp_instance_to_source_structure(instance)
    allowed = [set(table) for table in instance.template.relation_tables]
    direct_solutions = tuple(
        assignment
        for assignment in product(
            range(instance.template.carrier_size), repeat=instance.variable_count
        )
        if all(
            tuple(assignment[variable] for variable in constraint.scope)
            in allowed[
                next(
                    index
                    for index, symbol in enumerate(instance.template.signature)
                    if symbol.symbol_id == constraint.symbol_id
                )
            ]
            for constraint in instance.constraints
        )
    )
    homomorphisms = count_homomorphisms(source, instance.template)
    assert homomorphisms.count == len(direct_solutions)
    assert homomorphisms.count == 3


def test_rejects_duplicate_constraint_ids_unknown_symbols_and_bad_scopes() -> None:
    payload = _instance().model_dump()
    payload["constraints"][1]["constraint_id"] = "c0"
    with pytest.raises(ValidationError, match="constraint IDs must be unique"):
        FiniteCspInstance.model_validate(payload)

    payload = _instance().model_dump()
    payload["constraints"][0]["symbol_id"] = "Missing"
    with pytest.raises(ValidationError, match="must name a template relation"):
        FiniteCspInstance.model_validate(payload)

    payload = _instance().model_dump()
    payload["constraints"][0]["scope"] = [0, 3]
    with pytest.raises(ValidationError, match="declared axis"):
        FiniteCspInstance.model_validate(payload)


def test_forged_oversized_instance_is_rejected_before_recursive_copy() -> None:
    template = FiniteRelationalStructure(
        carrier_size=1,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=0),),
        relation_tables=(((),),),
    )
    forged = FiniteCspInstance.model_construct(
        template=template,
        variable_count=0,
        constraints=({"constraint_id": "c0", "symbol_id": "E", "scope": ()},) * 4_097,
    )
    with pytest.raises(OperationDomainValidationError) as error:
        csp_instance_to_source_structure(forged)
    assert error.value.errors()[0]["type"] == "relational.csp.instance_shape"


def test_catalog_operation_is_published_with_direct_structure_result() -> None:
    operation_id = "csp.instance.to_source_structure.compute"
    descriptor = Catalog.open().inspect(operation_id)
    assert descriptor.operation_id == operation_id
    assert any(tool.operation_id == operation_id for tool in BUILTIN_TOOLS)
    payload = json.loads(_instance().model_dump_json())
    result = invoke_operation(operation_id, payload, Catalog.open())
    assert FiniteRelationalStructure.model_validate(result.output) == (
        csp_instance_to_source_structure(_instance())
    )
