"""Native kernels consume the exact bounded snapshot checked by admission."""

from collections.abc import Iterator, Sequence
from typing import overload

import pytest

from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    check_polymorphism,
    induced_substructure,
)


class OnceSequence(Sequence[int]):
    def __init__(self, values: tuple[int, ...], change: bool) -> None:
        self.values = values
        self.change = change
        self.reads = 0

    def __len__(self) -> int:
        return len(self.values)

    @overload
    def __getitem__(self, key: int) -> int: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[int, ...]: ...

    def __getitem__(self, key: int | slice) -> int | tuple[int, ...]:
        return self.values[key]

    def __iter__(self) -> Iterator[int]:
        self.reads += 1
        if self.reads > 1:
            if self.change:
                return iter((999,))
            raise AssertionError("caller sequence iterated again after admission")
        return iter(self.values)


@pytest.mark.parametrize("change", [False, True])
def test_polymorphism_uses_only_admitted_snapshot(change: bool) -> None:
    source = FiniteRelationalStructure(carrier_size=2, signature=(), relation_tables=())
    table = OnceSequence((0, 0, 1, 1), change)
    result = check_polymorphism(source, 2, table)
    assert table.reads == 1
    assert result.polymorphism is not None
    assert result.polymorphism.operation_table == (0, 0, 1, 1)


@pytest.mark.parametrize("change", [False, True])
def test_induced_structure_uses_only_admitted_snapshot(change: bool) -> None:
    source = FiniteRelationalStructure(
        carrier_size=3,
        signature=(FiniteRelationSymbol(symbol_id="R", arity=2),),
        relation_tables=(((0, 1), (1, 2), (2, 0)),),
    )
    inclusion = OnceSequence((2, 0), change)
    result = induced_substructure(source, inclusion)
    assert inclusion.reads == 1
    assert result.inclusion == (2, 0)
    assert result.substructure.relation_tables == (((0, 1),),)
