"""Signature reduct of a finite relational structure."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    reduct_structure,
)
from jacobian.math.logic.relational_structures._admission import (
    MAX_RELATIONAL_REDUCT_WORK,
    admit_relational_reduct,
)
from jacobian.math.logic.relational_structures._models import RelationalReductRequest

OPERATION_ID = "relational_structure.reduct.compute"


def _structure(
    signature: tuple[tuple[str, int], ...],
    tables: tuple[tuple[tuple[int, ...], ...], ...],
    carrier_size: int,
) -> FiniteRelationalStructure:
    from jacobian.math.logic.relational_structures import FiniteRelationSymbol

    return FiniteRelationalStructure(
        carrier_size=carrier_size,
        signature=tuple(
            FiniteRelationSymbol(symbol_id=symbol_id, arity=arity)
            for symbol_id, arity in signature
        ),
        relation_tables=tables,
    )


# A binary relation E and a unary P over a two-element carrier.
TWO_SYMBOL = _structure(
    (("E", 2), ("P", 1)),
    (((0, 1),), ((1,),)),
    2,
)


def test_reduct_keeps_the_carrier_and_copies_the_selected_table() -> None:
    result = reduct_structure(TWO_SYMBOL, ("P",))
    assert result.reduct.carrier_size == TWO_SYMBOL.carrier_size
    assert [symbol.symbol_id for symbol in result.reduct.signature] == ["P"]
    assert result.reduct.relation_tables == (((1,),),)
    assert result.source_symbol_indices == (1,)


def test_reduct_follows_source_signature_order_not_selection_order() -> None:
    """The reduct axis is canonical in source order, whatever the input order."""
    forward = reduct_structure(TWO_SYMBOL, ("P", "E"))
    reversed_selection = reduct_structure(TWO_SYMBOL, ("E", "P"))
    for result in (forward, reversed_selection):
        assert [symbol.symbol_id for symbol in result.reduct.signature] == ["E", "P"]
        assert result.source_symbol_indices == (0, 1)
    assert forward.reduct == reversed_selection.reduct


def test_reduct_preserves_the_source_structure_verbatim() -> None:
    result = reduct_structure(TWO_SYMBOL, ("P",))
    assert result.source == TWO_SYMBOL


def test_request_rejects_duplicate_and_unknown_symbol_ids() -> None:
    with pytest.raises(ValidationError) as duplicate:
        RelationalReductRequest(source=TWO_SYMBOL, symbol_ids=("P", "P"))
    assert any("not_unique" in item["type"] for item in duplicate.value.errors())

    with pytest.raises(ValidationError) as unknown:
        RelationalReductRequest(source=TWO_SYMBOL, symbol_ids=("Q",))
    assert any("unknown" in item["type"] for item in unknown.value.errors())


def test_admission_rejects_duplicate_symbol_ids_before_construction() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        admit_relational_reduct(TWO_SYMBOL, ("P", "P"))
    assert error.value.errors()[0]["type"] == "relational.reduct.symbol_ids_not_unique"


def test_admission_returns_source_order_indices_and_copy_work() -> None:
    indices, work = admit_relational_reduct(TWO_SYMBOL, ("P", "E"))
    assert indices == (0, 1)
    # Two symbols: the binary E table of one row (1 + 1*(2+1) = 4) and the
    # unary P table of one row (1 + 1*(1+1) = 3).
    assert work == 7
    assert work <= MAX_RELATIONAL_REDUCT_WORK


def test_operation_is_published_in_the_catalog() -> None:
    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in ids
