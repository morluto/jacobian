"""Complete fixed-arity polymorphism families on finite structures."""

from __future__ import annotations

import json
from itertools import product

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    RelationalPolymorphismEnumerationRequest,
    RelationalPolymorphismFamily,
    enumerate_polymorphisms,
    operations,
)


def _binary_relation(mask: int) -> FiniteRelationalStructure:
    rows = tuple(product(range(2), repeat=2))
    return FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="R", arity=2),),
        relation_tables=(tuple(row for bit, row in enumerate(rows) if mask & (1 << bit)),),
    )


def _is_polymorphism(
    source: FiniteRelationalStructure, arity: int, operation_table: tuple[int, ...]
) -> bool:
    carrier_size = source.carrier_size
    input_tuples = tuple(product(range(carrier_size), repeat=arity))
    for symbol, relation in zip(source.signature, source.relation_tables, strict=True):
        relation_set = set(relation)
        for rows in product(relation, repeat=arity):
            output = tuple(
                operation_table[
                    next(
                        index
                        for index, candidate in enumerate(input_tuples)
                        if candidate == tuple(row[column] for row in rows)
                    )
                ]
                for column in range(symbol.arity)
            )
            if output not in relation_set:
                return False
    return True


def test_all_binary_relations_return_exact_complete_binary_families() -> None:
    # Independent brute-force oracle: all 16 binary relations on a two-point
    # carrier, checked against all 16 binary operation tables.
    for mask in range(16):
        source = _binary_relation(mask)
        expected = tuple(
            table
            for table in product(range(2), repeat=4)
            if _is_polymorphism(source, 2, table)
        )

        result = enumerate_polymorphisms(
            RelationalPolymorphismEnumerationRequest(source=source, arity=2)
        )

        assert result.operation_tables == expected
        assert result.source == source
        assert result.arity == 2


def test_unary_singleton_example_roundtrips_as_a_source_bound_value() -> None:
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="P", arity=1),),
        relation_tables=(((0,),),),
    )
    request = RelationalPolymorphismEnumerationRequest(source=source, arity=1)

    result = enumerate_polymorphisms(request)

    assert result.operation_tables == ((0, 0), (0, 1))
    restored = RelationalPolymorphismFamily.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.source == source


def test_empty_signature_three_element_binary_family_is_complete() -> None:
    result = enumerate_polymorphisms(
        RelationalPolymorphismEnumerationRequest(
            source=FiniteRelationalStructure(carrier_size=3), arity=2
        )
    )

    assert len(result.operation_tables) == 3**9
    assert result.operation_tables[0] == (0,) * 9
    assert result.operation_tables[-1] == (2,) * 9


def test_empty_carrier_has_one_empty_operation_and_checks_nullary_relations() -> None:
    source = FiniteRelationalStructure(
        carrier_size=0,
        signature=(
            FiniteRelationSymbol(symbol_id="E", arity=1),
            FiniteRelationSymbol(symbol_id="T", arity=0),
            FiniteRelationSymbol(symbol_id="F", arity=0),
        ),
        relation_tables=((), ((),), ()),
    )

    result = enumerate_polymorphisms(
        RelationalPolymorphismEnumerationRequest(source=source, arity=3)
    )

    assert result.operation_tables == ((),)


def test_candidate_and_work_refusals_precede_function_space_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_candidate_iteration(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("candidate function tables were requested before admission")

    monkeypatch.setattr(operations, "_candidate_operation_tables", no_candidate_iteration)

    too_many_candidates = FiniteRelationalStructure(carrier_size=4)
    with pytest.raises(OperationResourceAdmissionError, match="complete function space"):
        enumerate_polymorphisms(
            RelationalPolymorphismEnumerationRequest(
                source=too_many_candidates, arity=2
            )
        )

    full_binary = tuple(product(range(3), repeat=2))
    too_much_work = FiniteRelationalStructure(
        carrier_size=3,
        signature=(
            FiniteRelationSymbol(symbol_id="R", arity=2),
            FiniteRelationSymbol(symbol_id="S", arity=2),
        ),
        relation_tables=(full_binary, full_binary),
    )
    with pytest.raises(OperationResourceAdmissionError, match="family enumeration"):
        enumerate_polymorphisms(
            RelationalPolymorphismEnumerationRequest(source=too_much_work, arity=2)
        )


def test_operation_catalog_example_and_serialized_call() -> None:
    operation_id = "relational.polymorphisms.arity.enumerate"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    request = RelationalPolymorphismEnumerationRequest.model_validate(
        tool.examples[0].input
    )
    direct = tool.run(request)
    assert direct.operation_tables == ((0, 0), (0, 1))

    output = invoke_operation(
        operation_id,
        json.loads(request.model_dump_json()),
        Catalog.open(),
    ).output
    decoded = RelationalPolymorphismFamily.model_validate(output)
    assert decoded == direct
