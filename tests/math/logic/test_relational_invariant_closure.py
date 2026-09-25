"""Exact finite generated relations under supplied polymorphisms."""

from __future__ import annotations

import json
from itertools import product

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    RelationalInvariantClosure,
    RelationalInvariantClosureRequest,
    RelationalPolymorphism,
    close_relation_under_polymorphisms,
    operations,
)


def _index(arguments: tuple[int, ...], carrier_size: int) -> int:
    result = 0
    for value in arguments:
        result = result * carrier_size + value
    return result


def _oracle_closure(
    carrier_size: int,
    relation_arity: int,
    generators: tuple[tuple[int, ...], ...],
    operations: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[tuple[int, ...], ...]:
    """Naive full-product fixed point, independent of the queue kernel."""

    relation = set(generators)
    changed = True
    while changed:
        before = len(relation)
        for arity, table in operations:
            for arguments in product(sorted(relation), repeat=arity):
                image = tuple(
                    table[
                        _index(
                            tuple(argument[coordinate] for argument in arguments),
                            carrier_size,
                        )
                    ]
                    for coordinate in range(relation_arity)
                )
                relation.add(image)
        changed = len(relation) != before
    return tuple(sorted(relation))


def _operation(
    source: FiniteRelationalStructure, arity: int, table: tuple[int, ...]
) -> RelationalPolymorphism:
    return RelationalPolymorphism(source=source, arity=arity, operation_table=table)


def test_generated_relation_matches_full_product_fixed_point_oracle() -> None:
    # Exhaust every unary and binary operation on a two-point carrier over
    # every seed relation in arity one and two.
    source = FiniteRelationalStructure(carrier_size=2)
    for arity, relation_arity in ((1, 1), (2, 1), (2, 2)):
        universe = tuple(product(range(2), repeat=relation_arity))
        for table in product(range(2), repeat=2**arity):
            for mask in range(1 << len(universe)):
                seeds = tuple(
                    row for bit, row in enumerate(universe) if mask & (1 << bit)
                )
                request = RelationalInvariantClosureRequest(
                    source=source,
                    relation_arity=relation_arity,
                    generator_tuples=seeds,
                    polymorphisms=(_operation(source, arity, table),),
                )
                result = close_relation_under_polymorphisms(request)
                assert result.tuples == _oracle_closure(
                    source.carrier_size,
                    relation_arity,
                    seeds,
                    ((arity, table),),
                )
                assert result.generator_tuples == seeds


def test_closure_is_exact_on_nonempty_source_relations() -> None:
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=(
            {"symbol_id": "E", "arity": 2},
            {"symbol_id": "P", "arity": 1},
        ),
        relation_tables=(((0, 0), (1, 1)), ((0,), (1,))),
    )
    first_projection = _operation(source, 2, (0, 0, 1, 1))
    request = RelationalInvariantClosureRequest(
        source=source,
        relation_arity=2,
        generator_tuples=((0, 1),),
        polymorphisms=(first_projection,),
    )

    result = close_relation_under_polymorphisms(request)

    assert result.tuples == ((0, 1),)
    assert result.source == source


def test_two_supplied_operations_generate_the_least_common_closed_relation() -> None:
    source = FiniteRelationalStructure(carrier_size=2)
    constant_zero = _operation(source, 1, (0, 0))
    constant_one = _operation(source, 1, (1, 1))
    result = close_relation_under_polymorphisms(
        RelationalInvariantClosureRequest(
            source=source,
            relation_arity=1,
            generator_tuples=((1,), (0,), (0,)),
            polymorphisms=(constant_one, constant_zero, constant_zero),
        )
    )
    assert result.tuples == ((0,), (1,))


def test_non_polymorphism_is_rejected_before_it_can_claim_invariance() -> None:
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=({"symbol_id": "P", "arity": 1},),
        relation_tables=(((0,),),),
    )
    constant_one = _operation(source, 1, (1, 1))
    request = RelationalInvariantClosureRequest(
        source=source,
        relation_arity=1,
        generator_tuples=((0,),),
        polymorphisms=(constant_one,),
    )

    with pytest.raises(
        OperationDomainValidationError, match="fails to preserve relation P"
    ):
        close_relation_under_polymorphisms(request)


def test_empty_and_nullary_relations_keep_exact_power_semantics() -> None:
    source = FiniteRelationalStructure(carrier_size=0)
    empty_operation = _operation(source, 2, ())
    positive_arity = close_relation_under_polymorphisms(
        RelationalInvariantClosureRequest(
            source=source,
            relation_arity=1,
            generator_tuples=(),
            polymorphisms=(empty_operation,),
        )
    )
    assert positive_arity.tuples == ()

    nullary = close_relation_under_polymorphisms(
        RelationalInvariantClosureRequest(
            source=source,
            relation_arity=0,
            generator_tuples=((),),
            polymorphisms=(empty_operation,),
        )
    )
    assert nullary.tuples == ((),)


def test_admission_refuses_large_power_before_product_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FiniteRelationalStructure(carrier_size=64)
    unary = _operation(source, 1, (0,) * 64)
    request = RelationalInvariantClosureRequest(
        source=source,
        relation_arity=3,
        generator_tuples=(),
        polymorphisms=(unary,),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("tuple products were expanded before admission")

    monkeypatch.setattr(operations, "product", forbidden)
    with pytest.raises(OperationResourceAdmissionError, match="generated relation"):
        close_relation_under_polymorphisms(request)


def test_source_bound_value_roundtrips_and_catalog_call_agrees() -> None:
    operation_id = "relation.closure_under_operations.compute"
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    direct = tool.run(tool.request_type.model_validate(tool.examples[0].input))
    restored = RelationalInvariantClosure.model_validate_json(direct.model_dump_json())
    assert restored == direct
    assert direct.tuples == ((0,), (1,))

    output = invoke_operation(
        operation_id,
        json.loads(
            tool.request_type.model_validate(tool.examples[0].input).model_dump_json()
        ),
        Catalog.open(),
    ).output
    assert RelationalInvariantClosure.model_validate(output) == direct
