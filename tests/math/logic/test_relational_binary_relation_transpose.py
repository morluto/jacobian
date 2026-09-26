"""Independent exact checks for finite binary relation transposition."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
)
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    HomomorphismStatus,
    check_homomorphism,
    transpose_binary_relation,
)
from jacobian.math.logic.relational_structures._admission import (
    admit_binary_relation_transpose,
)
from jacobian.math.logic.relational_structures._models import (
    BinaryRelationTransposeRequest,
)
from jacobian.math.logic.relational_structures._tools import TOOLS


def _structure(
    carrier_size: int,
    relation: tuple[tuple[int, int], ...],
) -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=carrier_size,
        signature=(
            FiniteRelationSymbol(symbol_id="E", arity=2),
            FiniteRelationSymbol(symbol_id="P", arity=1),
            FiniteRelationSymbol(symbol_id="T", arity=0),
        ),
        relation_tables=(relation, (((1,),) if carrier_size > 1 else ()), ((),)),
    )


def test_all_binary_relations_on_three_points_match_independent_oracle() -> None:
    pairs: tuple[tuple[int, int], ...] = tuple(
        (left, right) for left in range(3) for right in range(3)
    )
    for mask in range(1 << len(pairs)):
        relation = tuple(
            pair for index, pair in enumerate(pairs) if mask & (1 << index)
        )
        source = _structure(3, relation)

        result = transpose_binary_relation(source, "E")
        oracle = tuple(sorted((right, left) for left, right in relation))

        assert result.relation_tables[0] == oracle
        assert result.signature == source.signature
        assert result.relation_tables[1:] == source.relation_tables[1:]
        assert transpose_binary_relation(result, "E") == source


def test_degenerate_relation_tables_and_serialized_composition() -> None:
    source = _structure(0, ())
    result = transpose_binary_relation(source, "E")
    restored = FiniteRelationalStructure.model_validate_json(result.model_dump_json())

    assert restored == source
    assert (
        check_homomorphism(
            restored, restored, tuple(range(restored.carrier_size))
        ).status
        is HomomorphismStatus.HOMOMORPHISM
    )


def test_native_kernel_rejects_unknown_and_nonbinary_symbols() -> None:
    source = _structure(3, ())
    with pytest.raises(OperationDomainValidationError) as unknown:
        transpose_binary_relation(source, "missing")
    assert (
        unknown.value.errors()[0]["type"]
        == "relational.structure.transpose_symbol_unknown"
    )

    with pytest.raises(OperationDomainValidationError) as unary:
        transpose_binary_relation(source, "P")
    assert unary.value.errors()[0]["type"] == "relational.structure.transpose_arity"


def test_admission_covers_complete_output_reconstruction() -> None:
    source = _structure(3, ((0, 1), (1, 2)))
    work = admit_binary_relation_transpose(source, 0)
    expected = sum(
        1 + len(table) * (symbol.arity + 1)
        for symbol, table in zip(source.signature, source.relation_tables, strict=True)
    ) + 2 * len(source.relation_tables[0])
    assert work == expected

    # Full-carrier accepted case: every pair on 64 points transposes
    # without refusal; the visit count is the exact reconstruction cost.
    full_relation = tuple((x, y) for x in range(64) for y in range(64))
    full_source = FiniteRelationalStructure(
        carrier_size=64,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(full_relation,),
    )
    full_work = admit_binary_relation_transpose(full_source, 0)
    assert full_work == 1 + len(full_relation) * 3 + 2 * len(full_relation)
    result = transpose_binary_relation(full_source, "E")
    assert result.relation_tables[0] == tuple(sorted((y, x) for x, y in full_relation))


def test_request_envelope_checks_selected_symbol_before_execution() -> None:
    source = _structure(3, ())
    with pytest.raises(ValidationError) as wrong_arity:
        BinaryRelationTransposeRequest(source=source, symbol_id="P")
    assert (
        wrong_arity.value.errors()[0]["type"] == "relational.structure.transpose_arity"
    )

    with pytest.raises(ValidationError) as unknown:
        BinaryRelationTransposeRequest(source=source, symbol_id="missing")
    assert unknown.value.errors()[0]["type"] == "relational.structure.transpose_symbol"


def test_transpose_remains_native_only_without_catalog_entry() -> None:
    source = _structure(3, ((0, 1), (1, 2)))
    assert transpose_binary_relation(source, "E").relation_tables[0] == ((1, 0), (2, 1))
    assert all(
        tool.operation_id != "relational_structure.transpose_binary_relation.compute"
        for tool in TOOLS
    )
