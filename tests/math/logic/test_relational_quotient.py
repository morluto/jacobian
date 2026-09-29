"""Exact contract tests for quotients of finite relational structures."""

from __future__ import annotations

import itertools

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    quotient_structure,
)


def _brute_quotient(source, partition):
    """Independent truth-table definition over every quotient tuple cell."""
    blocks = sorted(
        set(partition),
        key=lambda label: min(i for i, value in enumerate(partition) if value == label),
    )
    index = {label: i for i, label in enumerate(blocks)}
    projection = tuple(index[label] for label in partition)
    tables = []
    for symbol, source_table in zip(
        source.signature, source.relation_tables, strict=True
    ):
        members = [
            tuple(i for i, image in enumerate(projection) if image == q)
            for q in range(len(blocks))
        ]
        quotient_rows = []
        for row in itertools.product(range(len(blocks)), repeat=symbol.arity):
            fiber = tuple(itertools.product(*(members[q] for q in row)))
            truth = {tuple(source_row) in source_table for source_row in fiber}
            assert len(truth) == 1
            if truth == {True}:
                quotient_rows.append(row)
        tables.append(tuple(quotient_rows))
    return projection, tuple(tables)


def test_complete_relation_fibers_form_the_exact_quotient() -> None:
    source = FiniteRelationalStructure(
        carrier_size=4,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(((0, 2), (0, 3), (1, 2), (1, 3)),),
    )
    result = quotient_structure(source, (91, 91, -8, -8))

    projection, tables = _brute_quotient(source, (91, 91, -8, -8))
    assert result.quotient_map == projection == (0, 0, 1, 1)
    assert result.quotient.carrier_size == 2
    assert result.quotient.relation_tables == tables == (((0, 1),),)
    assert result.model_validate_json(result.model_dump_json()) == result


def test_all_relation_symbols_and_nullary_truth_are_preserved() -> None:
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=(
            FiniteRelationSymbol(symbol_id="P", arity=1),
            FiniteRelationSymbol(symbol_id="T", arity=0),
        ),
        relation_tables=(((0,), (1,)), ((),)),
    )
    result = quotient_structure(source, (4, 4))
    assert result.quotient.relation_tables == (((0,),), ((),))


def test_partial_relation_fiber_is_not_a_congruence() -> None:
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(((0, 1),),),
    )
    with pytest.raises(OperationDomainValidationError, match="partial source fiber"):
        quotient_structure(source, (0, 0))


def test_partition_axis_and_exact_integer_labels_are_required() -> None:
    source = FiniteRelationalStructure(carrier_size=1)
    with pytest.raises(OperationDomainValidationError, match="one label per source"):
        quotient_structure(source, ())
    with pytest.raises(OperationDomainValidationError, match="exact integer"):
        quotient_structure(source, (True,))


def test_empty_carrier_has_empty_quotient_map() -> None:
    source = FiniteRelationalStructure(carrier_size=0)
    result = quotient_structure(source, ())
    assert result.quotient.carrier_size == 0
    assert result.quotient_map == ()
