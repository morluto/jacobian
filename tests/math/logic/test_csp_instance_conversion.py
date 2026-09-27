"""Exact CSP-to-relational-structure conversion contracts."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
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


def test_missing_nested_template_fields_use_typed_domain_error() -> None:
    malformed_template = FiniteRelationalStructure.model_construct(
        signature=(), relation_tables=()
    )
    forged = FiniteCspInstance.model_construct(
        template=malformed_template, variable_count=0, constraints=()
    )
    with pytest.raises(OperationDomainValidationError) as error:
        csp_instance_to_source_structure(forged)
    assert error.value.errors()[0]["type"] == "relational.csp.instance_shape"

    malformed_symbol = FiniteRelationSymbol.model_construct(symbol_id="R")
    malformed_template = FiniteRelationalStructure.model_construct(
        carrier_size=1, signature=(malformed_symbol,), relation_tables=((),)
    )
    forged = FiniteCspInstance.model_construct(
        template=malformed_template, variable_count=0, constraints=()
    )
    with pytest.raises(OperationDomainValidationError) as error:
        csp_instance_to_source_structure(forged)
    assert error.value.errors()[0]["type"] == "relational.csp.instance_shape"


def test_conversion_is_not_limited_by_unrelated_transport_row_cap() -> None:
    symbol_count = 5
    symbols = tuple(
        FiniteRelationSymbol(symbol_id=f"R{index}", arity=4)
        for index in range(symbol_count)
    )
    table = tuple(product(range(8), repeat=4))
    instance = FiniteCspInstance(
        template=FiniteRelationalStructure(
            carrier_size=8,
            signature=symbols,
            relation_tables=(table,) * symbol_count,
        ),
        variable_count=0,
        constraints=(),
    )
    converted = csp_instance_to_source_structure(instance)
    assert converted.relation_tables == ((),) * symbol_count
