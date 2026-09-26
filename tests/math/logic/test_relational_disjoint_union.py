"""Definition-level checks for finite relational disjoint unions."""

from __future__ import annotations

from itertools import islice, product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    HomomorphismStatus,
    RelationalDisjointUnionResult,
    check_homomorphism,
    disjoint_union_structure,
)
from jacobian.math.logic.relational_structures._models import (
    RelationalDisjointUnionRequest,
)
from jacobian.math.logic.relational_structures._tools import TOOLS

_SIGNATURE = (
    FiniteRelationSymbol(symbol_id="E", arity=2),
    FiniteRelationSymbol(symbol_id="P", arity=1),
    FiniteRelationSymbol(symbol_id="T", arity=0),
)


def _structure(
    carrier_size: int,
    edges: tuple[tuple[int, int], ...],
    points: tuple[tuple[int, ...], ...],
    truth: bool,
) -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=carrier_size,
        signature=_SIGNATURE,
        relation_tables=(edges, points, ((),) if truth else ()),
    )


def _contains(
    structure: FiniteRelationalStructure, symbol_index: int, row: tuple[int, ...]
) -> bool:
    return row in structure.relation_tables[symbol_index]


def _homomorphisms(
    source: FiniteRelationalStructure, target: FiniteRelationalStructure
) -> set[tuple[int, ...]]:
    """Enumerate total maps and check only the defining relation condition."""

    maps = product(range(target.carrier_size), repeat=source.carrier_size)
    accepted = set()
    for carrier_map in maps:
        if source.signature != target.signature:
            continue
        preserves = all(
            tuple(carrier_map[label] for label in row) in target_table
            for source_table, target_table in zip(
                source.relation_tables, target.relation_tables, strict=True
            )
            for row in source_table
        )
        if preserves:
            accepted.add(tuple(carrier_map))
    return accepted


def test_union_tables_follow_component_membership_and_exclude_mixed_rows() -> None:
    left = _structure(2, ((0, 1),), ((1,),), False)
    right = _structure(2, ((0, 0),), (), True)

    result = disjoint_union_structure(left, right)
    union = result.disjoint_union

    assert union.carrier_size == 4
    assert union.signature == _SIGNATURE
    assert result.left_inclusion.mapping == (0, 1)
    assert result.right_inclusion.mapping == (2, 3)
    assert union.relation_tables == (((0, 1), (2, 2)), ((1,),), ((),))

    edge_rows = set(union.relation_tables[0])
    for first, second in product(range(4), repeat=2):
        if first < 2 and second < 2:
            expected = _contains(left, 0, (first, second))
        elif first >= 2 and second >= 2:
            expected = _contains(right, 0, (first - 2, second - 2))
        else:
            expected = False
        assert ((first, second) in edge_rows) is expected

    point_rows = set(union.relation_tables[1])
    for point in range(4):
        expected = (
            _contains(left, 1, (point,))
            if point < 2
            else _contains(right, 1, (point - 2,))
        )
        assert ((point,) in point_rows) is expected


def test_inclusions_and_relation_transport_survive_serialization() -> None:
    left = _structure(2, ((0, 1),), ((1,),), True)
    right = _structure(1, ((0, 0),), ((0,),), False)
    result = disjoint_union_structure(left, right)
    restored = RelationalDisjointUnionResult.model_validate_json(
        result.model_dump_json()
    )

    assert restored == result
    for component, inclusion in (
        (restored.left, restored.left_inclusion.mapping),
        (restored.right, restored.right_inclusion.mapping),
    ):
        checked = check_homomorphism(component, restored.disjoint_union, inclusion)
        assert checked.status is HomomorphismStatus.HOMOMORPHISM
        for symbol_index, symbol in enumerate(component.signature):
            candidates = product(range(component.carrier_size), repeat=symbol.arity)
            union_rows = set(restored.disjoint_union.relation_tables[symbol_index])
            for row in candidates:
                image = tuple(inclusion[label] for label in row)
                if symbol.arity == 0:
                    assert row not in component.relation_tables[symbol_index] or (
                        image in union_rows
                    )
                else:
                    assert (row in component.relation_tables[symbol_index]) == (
                        image in union_rows
                    )
    assert () not in restored.right.relation_tables[2]
    assert () in restored.disjoint_union.relation_tables[2]


def test_homomorphisms_from_union_are_pairs_of_component_homomorphisms() -> None:
    left = _structure(2, ((0, 1),), ((1,),), False)
    right = _structure(1, ((0, 0),), (), True)
    result = disjoint_union_structure(left, right)
    target = _structure(2, ((0, 1), (1, 1)), ((1,),), True)

    expected_pairs = set(
        product(_homomorphisms(left, target), _homomorphisms(right, target))
    )
    actual_pairs = {
        (
            tuple(carrier_map[label] for label in result.left_inclusion.mapping),
            tuple(carrier_map[label] for label in result.right_inclusion.mapping),
        )
        for carrier_map in _homomorphisms(result.disjoint_union, target)
    }

    assert actual_pairs == expected_pairs


def test_empty_carriers_relations_and_nullary_truth_obey_sum_semantics() -> None:
    empty_false = _structure(0, (), (), False)
    empty_true = _structure(0, (), (), True)

    both_false = disjoint_union_structure(empty_false, empty_false)
    one_true = disjoint_union_structure(empty_false, empty_true)

    assert both_false.disjoint_union.carrier_size == 0
    assert both_false.disjoint_union.relation_tables == ((), (), ())
    assert both_false.left_inclusion.mapping == both_false.right_inclusion.mapping == ()
    assert one_true.disjoint_union.carrier_size == 0
    assert one_true.disjoint_union.relation_tables == ((), (), ((),))

    bare_empty = FiniteRelationalStructure(carrier_size=0)
    bare_two = FiniteRelationalStructure(carrier_size=2)
    bare_union = disjoint_union_structure(bare_empty, bare_two)
    assert bare_union.disjoint_union.signature == ()
    assert bare_union.disjoint_union.carrier_size == 2
    assert bare_union.left_inclusion.mapping == ()
    assert bare_union.right_inclusion.mapping == (0, 1)


def test_catalog_example_and_request_are_wired() -> None:
    tool = next(
        entry
        for entry in TOOLS
        if entry.operation_id == "relational_structure.disjoint_union.compute"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    assert isinstance(request, RelationalDisjointUnionRequest)
    result = tool.run(request)
    assert isinstance(result, RelationalDisjointUnionResult)
    assert result.disjoint_union.relation_tables == (
        ((0, 1), (2, 2)),
        ((1,),),
        ((),),
    )


def test_mismatched_signatures_and_unrepresentable_carrier_fail_typed() -> None:
    left = _structure(40, (), (), False)
    right = _structure(25, (), (), False)
    with pytest.raises(OperationResourceAdmissionError) as carrier_error:
        disjoint_union_structure(left, right)
    assert (
        carrier_error.value.errors()[0]["type"]
        == "relational.structure.disjoint_union_carrier_bound"
    )

    other_signature = FiniteRelationalStructure(
        carrier_size=1,
        signature=(FiniteRelationSymbol(symbol_id="Q", arity=2),),
        relation_tables=((),),
    )
    with pytest.raises(OperationDomainValidationError) as signature_error:
        disjoint_union_structure(_structure(1, (), (), False), other_signature)
    assert (
        signature_error.value.errors()[0]["type"]
        == "relational.structure.disjoint_union_signature"
    )


def test_union_relation_row_growth_is_admitted_before_construction() -> None:
    rows = tuple(islice(product(range(32), repeat=4), 3_000))
    signature = (FiniteRelationSymbol(symbol_id="Q", arity=4),)
    left = FiniteRelationalStructure(
        carrier_size=32, signature=signature, relation_tables=(rows[:3_000],)
    )
    right = FiniteRelationalStructure(
        carrier_size=32, signature=signature, relation_tables=(rows[:3_000],)
    )

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        disjoint_union_structure(left, right)
    assert (
        exc_info.value.errors()[0]["type"]
        == "relational.structure.disjoint_union_table_bound"
    )
