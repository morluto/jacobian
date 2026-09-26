"""Exact contract tests for finite relational structure homomorphism checking."""

from __future__ import annotations

import json
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    HomomorphismCheckResult,
    HomomorphismEnumerationResult,
    HomomorphismStatus,
    check_homomorphism,
    enumerate_homomorphisms,
)
from jacobian.math.logic.relational_structures._models import (
    HomomorphismCheckRequest,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_ARITY,
    MAX_RELATIONAL_CARRIER,
    MAX_RELATIONAL_SYMBOLS,
    MAX_RELATIONAL_TABLE_ROWS,
    MAX_RELATIONAL_TRANSPORT_TUPLES,
)

OPERATION_ID = "relational.homomorphism.check"
ENUMERATION_OPERATION_ID = "relational.homomorphism.enumerate.compute"

_EDGE = (FiniteRelationSymbol(symbol_id="E", arity=2),)


def _structure(
    carrier_size: int,
    signature: tuple[FiniteRelationSymbol, ...],
    tables: tuple[tuple[tuple[int, ...], ...], ...],
) -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=carrier_size, signature=signature, relation_tables=tables
    )


def _three_cycle() -> FiniteRelationalStructure:
    return _structure(3, _EDGE, (((0, 1), (1, 2), (2, 0)),))


def _directed_triangle() -> FiniteRelationalStructure:
    return _structure(
        3,
        _EDGE,
        (((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),),
    )


def _loop_point() -> FiniteRelationalStructure:
    return _structure(1, _EDGE, (((0, 0),),))


def _edgeless_two() -> FiniteRelationalStructure:
    return _structure(2, _EDGE, ((),))


def _catalog_tool():
    return next(tool for tool in BUILTIN_TOOLS if tool.operation_id == OPERATION_ID)


def _wire_payload(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: tuple[int, ...],
) -> dict:
    return json.loads(
        HomomorphismCheckRequest(
            source=source, target=target, carrier_map=carrier_map
        ).model_dump_json()
    )


def test_directed_three_cycle_into_triangle_is_valid() -> None:
    result = check_homomorphism(_three_cycle(), _directed_triangle(), (0, 1, 2))
    assert result.status is HomomorphismStatus.HOMOMORPHISM
    assert result.witness is None
    assert [
        (p.symbol_id, p.source_tuples, p.preserved_tuples)
        for p in result.symbol_profiles
    ] == [("E", 3, 3)]


def test_noninjective_map_to_loop_point_is_valid() -> None:
    """A noninjective homomorphism collapses the cycle onto one loop."""

    result = check_homomorphism(_three_cycle(), _loop_point(), (0, 0, 0))
    assert result.status is HomomorphismStatus.HOMOMORPHISM
    assert result.symbol_profiles[0].preserved_tuples == 3


def test_edge_into_antichain_yields_first_witness() -> None:
    """A 2-element antichain target admits no edge image."""

    source = _structure(2, _EDGE, (((0, 1),),))
    result = check_homomorphism(source, _edgeless_two(), (0, 1))
    assert result.status is HomomorphismStatus.NOT_HOMOMORPHISM
    assert result.witness is not None
    assert result.witness.symbol_id == "E"
    assert result.witness.source_tuple == (0, 1)
    assert result.witness.image_tuple == (0, 1)
    assert result.symbol_profiles[0].source_tuples == 1
    assert result.symbol_profiles[0].preserved_tuples == 0


def test_first_witness_follows_deterministic_row_order() -> None:
    """The witness is the first violating row, and counts stay complete."""

    source = _structure(3, _EDGE, (((0, 1), (1, 2), (2, 0)),))
    # Target keeps only the image of the first two rows under the identity.
    target = _structure(3, _EDGE, (((0, 1), (1, 2)),))
    result = check_homomorphism(source, target, (0, 1, 2))
    assert result.status is HomomorphismStatus.NOT_HOMOMORPHISM
    assert result.witness is not None
    assert result.witness.source_tuple == (2, 0)
    assert result.symbol_profiles[0].preserved_tuples == 2


def test_empty_signature_makes_every_map_a_homomorphism() -> None:
    source = _structure(4, (), ())
    target = _structure(2, (), ())
    result = check_homomorphism(source, target, (0, 1, 0, 1))
    assert result.status is HomomorphismStatus.HOMOMORPHISM
    assert result.symbol_profiles == ()
    assert result.witness is None


def test_identity_map_is_always_a_homomorphism() -> None:
    for structure in (_three_cycle(), _directed_triangle(), _loop_point()):
        result = check_homomorphism(
            structure, structure, tuple(range(structure.carrier_size))
        )
        assert result.status is HomomorphismStatus.HOMOMORPHISM


def test_composition_of_checked_homomorphisms_is_checked() -> None:
    """h: C3 -> K3 and g: K3 -> loop compose to a checked homomorphism."""

    cycle = _three_cycle()
    triangle = _directed_triangle()
    loop = _loop_point()
    first = check_homomorphism(cycle, triangle, (0, 1, 2))
    second = check_homomorphism(triangle, loop, (0, 0, 0))
    assert first.status is HomomorphismStatus.HOMOMORPHISM
    assert second.status is HomomorphismStatus.HOMOMORPHISM
    composed = tuple(second.carrier_map[image] for image in first.carrier_map)
    assert composed == (0, 0, 0)
    third = check_homomorphism(cycle, loop, composed)
    assert third.status is HomomorphismStatus.HOMOMORPHISM


def test_mixed_arities_replay_exhaustively_with_complete_counts() -> None:
    """Nullary, unary, and binary symbols each keep transport counts."""

    signature = (
        FiniteRelationSymbol(symbol_id="T", arity=0),
        FiniteRelationSymbol(symbol_id="U", arity=1),
        FiniteRelationSymbol(symbol_id="E", arity=2),
    )
    source = _structure(2, signature, (((),), ((0,), (1,)), ((0, 1), (1, 0))))
    # The nullary relation is false in the target; U drops one element.
    target = _structure(2, signature, ((), ((0,),), ((0, 0), (0, 1), (1, 0), (1, 1))))
    result = check_homomorphism(source, target, (0, 0))
    assert result.status is HomomorphismStatus.NOT_HOMOMORPHISM
    assert result.witness is not None
    # Signature order decides the first witness: the nullary symbol T.
    assert result.witness.symbol_id == "T"
    assert result.witness.source_tuple == ()
    assert result.witness.image_tuple == ()
    assert [
        (p.symbol_id, p.source_tuples, p.preserved_tuples)
        for p in result.symbol_profiles
    ] == [("T", 1, 0), ("U", 2, 2), ("E", 2, 2)]


def test_duplicate_rows_normalize_once_and_order_is_transport() -> None:
    shuffled = FiniteRelationalStructure(
        carrier_size=2,
        signature=_EDGE,
        relation_tables=[((1, 0), (0, 1), (1, 0), (0, 1))],
    )
    canonical = _structure(2, _EDGE, (((0, 1), (1, 0)),))
    assert shuffled == canonical
    assert shuffled.relation_tables == (((0, 1), (1, 0)),)


def test_forged_tuple_tables_are_rejected_structurally() -> None:
    with pytest.raises(ValidationError) as arity:
        _structure(2, _EDGE, (((0, 1, 1),),))
    assert (
        arity.value.errors(include_url=False)[0]["type"]
        == "relational.structure.tuple_arity"
    )
    with pytest.raises(ValidationError) as carrier:
        _structure(2, _EDGE, (((0, 5),),))
    assert (
        carrier.value.errors(include_url=False)[0]["type"]
        == "relational.structure.tuple_carrier"
    )
    with pytest.raises(ValidationError) as count:
        _structure(2, _EDGE, ())
    assert (
        count.value.errors(include_url=False)[0]["type"]
        == "relational.structure.table_count"
    )
    with pytest.raises(ValidationError) as identity:
        _structure(
            2,
            (
                FiniteRelationSymbol(symbol_id="E", arity=2),
                FiniteRelationSymbol(symbol_id="E", arity=1),
            ),
            ((), ()),
        )
    assert (
        identity.value.errors(include_url=False)[0]["type"]
        == "relational.structure.symbol_identity"
    )


def test_signature_mismatch_is_typed_rejected_on_both_paths() -> None:
    foreign = _structure(3, (FiniteRelationSymbol(symbol_id="F", arity=2),), ((),))
    with pytest.raises(OperationDomainValidationError) as native:
        check_homomorphism(_three_cycle(), foreign, (0, 1, 2))
    assert (
        native.value.errors()[0]["type"] == "relational.homomorphism.signature_mismatch"
    )
    payload = {
        "source": json.loads(_three_cycle().model_dump_json()),
        "target": json.loads(foreign.model_dump_json()),
        "carrier_map": [0, 1, 2],
    }
    with pytest.raises(ValidationError) as wire:
        HomomorphismCheckRequest.model_validate(payload)
    assert (
        wire.value.errors(include_url=False)[0]["type"]
        == "relational.homomorphism.signature_mismatch"
    )


def test_malformed_carrier_maps_are_boundary_invalid() -> None:
    with pytest.raises(OperationDomainValidationError) as short:
        check_homomorphism(_three_cycle(), _directed_triangle(), (0, 1))
    assert short.value.errors()[0]["type"] == "relational.homomorphism.carrier_map_axis"
    with pytest.raises(OperationDomainValidationError) as outside:
        check_homomorphism(_three_cycle(), _loop_point(), (0, 0, 1))
    assert (
        outside.value.errors()[0]["type"] == "relational.homomorphism.carrier_map_value"
    )
    with pytest.raises(ValidationError) as wire:
        HomomorphismCheckRequest.model_validate(
            _wire_payload(_three_cycle(), _directed_triangle(), (0, 1))
        )
    assert (
        wire.value.errors(include_url=False)[0]["type"]
        == "relational.homomorphism.carrier_map_axis"
    )


def test_native_and_catalog_paths_agree() -> None:
    tool = _catalog_tool()
    payload = _wire_payload(_three_cycle(), _directed_triangle(), (0, 1, 2))
    catalog_result = tool.run(tool.request_type.model_validate(payload))
    native_result = check_homomorphism(_three_cycle(), _directed_triangle(), (0, 1, 2))
    assert catalog_result == native_result

    negative_payload = _wire_payload(_directed_triangle(), _three_cycle(), (0, 1, 2))
    catalog_negative = tool.run(tool.request_type.model_validate(negative_payload))
    assert catalog_negative == check_homomorphism(
        _directed_triangle(), _three_cycle(), (0, 1, 2)
    )
    assert catalog_negative.status is HomomorphismStatus.NOT_HOMOMORPHISM


def test_catalog_example_executes() -> None:
    tool = _catalog_tool()
    for example in tool.examples:
        result = tool.run(
            tool.request_type.model_validate(json.loads(json.dumps(example.input)))
        )
        assert result.status is HomomorphismStatus.HOMOMORPHISM


def test_serialization_round_trips() -> None:
    structure = _three_cycle()
    assert (
        FiniteRelationalStructure.model_validate_json(structure.model_dump_json())
        == structure
    )
    for result in (
        check_homomorphism(structure, _directed_triangle(), (0, 1, 2)),
        check_homomorphism(_directed_triangle(), structure, (0, 1, 2)),
    ):
        assert (
            HomomorphismCheckResult.model_validate_json(result.model_dump_json())
            == result
        )


def test_envelope_accepts_full_carrier_and_rejects_above_it() -> None:
    empty_signature = _structure(MAX_RELATIONAL_CARRIER, (), ())
    result = check_homomorphism(
        empty_signature, empty_signature, tuple(range(MAX_RELATIONAL_CARRIER))
    )
    assert result.status is HomomorphismStatus.HOMOMORPHISM
    with pytest.raises(ValidationError):
        _structure(MAX_RELATIONAL_CARRIER + 1, (), ())


def test_envelope_rejects_excess_symbols_and_arity() -> None:
    symbols = tuple(
        FiniteRelationSymbol(symbol_id=f"S{index}", arity=1)
        for index in range(MAX_RELATIONAL_SYMBOLS + 1)
    )
    with pytest.raises(ValidationError):
        _structure(1, symbols, tuple(((),) for _ in symbols))
    with pytest.raises(ValidationError):
        FiniteRelationSymbol(symbol_id="R", arity=MAX_RELATIONAL_ARITY + 1)
    accepted = _structure(
        1,
        (FiniteRelationSymbol(symbol_id="R", arity=MAX_RELATIONAL_ARITY),),
        (((0, 0, 0, 0),),),
    )
    assert check_homomorphism(accepted, accepted, (0,)).status is (
        HomomorphismStatus.HOMOMORPHISM
    )


def test_envelope_rejects_oversized_tuple_table() -> None:
    pairs = tuple(
        (row // MAX_RELATIONAL_CARRIER, row % MAX_RELATIONAL_CARRIER)
        for row in range(MAX_RELATIONAL_TABLE_ROWS + 1)
    )
    with pytest.raises(ValidationError) as rows:
        _structure(MAX_RELATIONAL_CARRIER, _EDGE, (pairs,))
    assert (
        rows.value.errors(include_url=False)[0]["type"]
        == "relational.structure.table_rows"
    )


def test_transport_envelope_admits_boundary_and_rejects_above() -> None:
    """sum |R^A| is preflighted before any tuple is transported."""

    signature = tuple(
        FiniteRelationSymbol(symbol_id=f"S{index}", arity=2)
        for index in range(MAX_RELATIONAL_SYMBOLS)
    )
    per_table = MAX_RELATIONAL_TRANSPORT_TUPLES // MAX_RELATIONAL_SYMBOLS
    assert per_table <= MAX_RELATIONAL_TABLE_ROWS
    rows = tuple(
        (row // MAX_RELATIONAL_CARRIER, row % MAX_RELATIONAL_CARRIER)
        for row in range(per_table)
    )
    source = _structure(
        MAX_RELATIONAL_CARRIER,
        signature,
        tuple(rows for _ in signature),
    )
    target = source
    identity = tuple(range(MAX_RELATIONAL_CARRIER))
    boundary = check_homomorphism(source, target, identity)
    assert boundary.status is HomomorphismStatus.HOMOMORPHISM
    assert sum(p.source_tuples for p in boundary.symbol_profiles) == (
        MAX_RELATIONAL_TRANSPORT_TUPLES
    )

    fresh_row = (MAX_RELATIONAL_CARRIER - 1, MAX_RELATIONAL_CARRIER - 1)
    assert fresh_row not in rows
    over_rows = (*rows, fresh_row)
    over = _structure(
        MAX_RELATIONAL_CARRIER,
        signature,
        tuple(over_rows for _ in signature),
    )
    with pytest.raises(OperationResourceAdmissionError) as admission:
        check_homomorphism(over, target, identity)
    assert (
        admission.value.errors()[0]["type"] == "relational.homomorphism.transport_bound"
    )


def test_operation_is_published_in_the_catalog() -> None:
    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in ids


def test_homomorphism_enumeration_matches_independent_map_scan() -> None:
    source = _structure(2, _EDGE, (((0, 1),),))
    target = _three_cycle()
    result = enumerate_homomorphisms(source, target)
    independently_expected = tuple(
        mapping
        for mapping in product(range(3), repeat=2)
        if (mapping[0], mapping[1]) in set(target.relation_tables[0])
    )
    assert result.source == source
    assert result.target == target
    assert result.total_candidates == 9
    assert result.carrier_maps == independently_expected == ((0, 1), (1, 2), (2, 0))


def test_homomorphism_enumeration_handles_empty_source_and_no_maps() -> None:
    empty_source = _structure(0, _EDGE, ((),))
    result = enumerate_homomorphisms(empty_source, _three_cycle())
    assert result.carrier_maps == ((),)
    assert result.total_candidates == 1
    assert (
        enumerate_homomorphisms(
            _three_cycle(), _structure(0, _EDGE, ((),))
        ).carrier_maps
        == ()
    )


def test_homomorphism_enumeration_admits_before_candidate_product(monkeypatch) -> None:
    import jacobian.math.logic.relational_structures.operations as operations

    source = _structure(6, _EDGE, ((),))
    target = _structure(8, _EDGE, ((),))

    def forbidden_product(*args, **kwargs):
        raise AssertionError(
            "candidate product must not be constructed before admission"
        )

    monkeypatch.setattr(operations, "product", forbidden_product)
    with pytest.raises(OperationResourceAdmissionError):
        operations.enumerate_homomorphisms(source, target)


def test_enumeration_output_admission_bounds_retained_map_labels(
    monkeypatch,
) -> None:
    import jacobian.math.logic.relational_structures._admission as admission
    import jacobian.math.logic.relational_structures.operations as operations

    source = _structure(5, _EDGE, (((0, 1), (1, 2), (2, 3), (3, 4)),))
    target = _three_cycle()

    def forbidden_product(*_args, **_kwargs):
        raise AssertionError(
            "candidate maps must not be expanded before output admission"
        )

    monkeypatch.setattr(operations, "product", forbidden_product)
    monkeypatch.setattr(admission, "MAX_HOMOMORPHISM_ENUMERATION_MAP_LABELS", 100)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        operations.enumerate_homomorphisms(source, target)
    assert exc_info.value.errors()[0]["type"] == (
        "relational.homomorphism.enumeration_output"
    )


def test_homomorphism_enumeration_tool_example_executes() -> None:
    tool = next(
        tool for tool in BUILTIN_TOOLS if tool.operation_id == ENUMERATION_OPERATION_ID
    )
    example = tool.examples[0]
    result = tool.run(tool.request_type.model_validate(example.input))
    assert isinstance(result, HomomorphismEnumerationResult)
    assert result.carrier_maps == ((0, 1), (1, 2), (2, 0))


def test_homomorphism_enumeration_is_published() -> None:
    assert ENUMERATION_OPERATION_ID in {tool.operation_id for tool in BUILTIN_TOOLS}
