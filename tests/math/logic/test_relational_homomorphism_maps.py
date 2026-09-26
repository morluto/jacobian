"""Finite direct oracles for source-bound relational homomorphisms."""

from itertools import product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    RelationalCarrierMap,
    RelationalHomomorphism,
    compose_homomorphisms,
    homomorphism_identity,
)
from jacobian.math.logic.relational_structures._admission import (
    MAX_HOMOMORPHISM_COMPOSITION_TUPLE_REPLAYS,
)
from jacobian.math.logic.relational_structures._tools import TOOLS


def _structure(rows: tuple[tuple[int, ...], ...]) -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(rows,),
    )


def _direct_homomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    mapping: tuple[int, ...],
) -> bool:
    """Independent tuple-membership oracle for the preserving-map law."""

    target_rows = {
        symbol.symbol_id: set(table)
        for symbol, table in zip(target.signature, target.relation_tables, strict=True)
    }
    for symbol, table in zip(source.signature, source.relation_tables, strict=True):
        if any(
            tuple(mapping[coordinate] for coordinate in row)
            not in target_rows[symbol.symbol_id]
            for row in table
        ):
            return False
    return True


def _typed_map(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    mapping: tuple[int, ...],
) -> RelationalHomomorphism:
    """Make a structurally valid value; the consumer checks its claim."""

    return RelationalHomomorphism(source=source, target=target, mapping=mapping)


def test_identity_preserves_empty_and_nullary_relations_after_json_round_trip() -> None:
    false_nullary = FiniteRelationalStructure(
        carrier_size=0,
        signature=(FiniteRelationSymbol(symbol_id="F", arity=0),),
        relation_tables=((),),
    )
    true_nullary = FiniteRelationalStructure(
        carrier_size=0,
        signature=(FiniteRelationSymbol(symbol_id="T", arity=0),),
        relation_tables=(((),),),
    )

    for structure in (false_nullary, true_nullary):
        result = homomorphism_identity(structure)
        decoded = RelationalHomomorphism.model_validate_json(result.model_dump_json())
        assert decoded.source == structure
        assert decoded.target == structure
        assert decoded.mapping == ()
        assert decoded.injective and decoded.surjective
        assert _direct_homomorphism(decoded.source, decoded.target, decoded.mapping)


def test_composition_matches_exhaustive_direct_oracle() -> None:
    edge = FiniteRelationalStructure(
        carrier_size=2,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(((0, 1),),),
    )
    cycle = FiniteRelationalStructure(
        carrier_size=3,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(((0, 1), (1, 2), (2, 0)),),
    )
    triangle = FiniteRelationalStructure(
        carrier_size=3,
        signature=cycle.signature,
        relation_tables=(((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),),
    )
    first_maps = tuple(
        candidate
        for candidate in product(range(cycle.carrier_size), repeat=edge.carrier_size)
        if _direct_homomorphism(edge, cycle, candidate)
    )
    second_maps = tuple(
        candidate
        for candidate in product(
            range(triangle.carrier_size), repeat=cycle.carrier_size
        )
        if _direct_homomorphism(cycle, triangle, candidate)
    )
    assert len(first_maps) == 3
    assert len(second_maps) == 6

    for first_map in first_maps:
        for second_map in second_maps:
            result = compose_homomorphisms(
                _typed_map(edge, cycle, first_map),
                _typed_map(cycle, triangle, second_map),
            )
            expected = tuple(second_map[value] for value in first_map)
            decoded = RelationalHomomorphism.model_validate_json(
                result.model_dump_json()
            )
            assert decoded.mapping == expected
            assert decoded.source == edge and decoded.target == triangle
            assert _direct_homomorphism(edge, triangle, expected)


def test_composition_rejects_a_forged_nullary_preservation_claim() -> None:
    true_relation = FiniteRelationalStructure(
        carrier_size=0,
        signature=(FiniteRelationSymbol(symbol_id="T", arity=0),),
        relation_tables=(((),),),
    )
    false_relation = FiniteRelationalStructure(
        carrier_size=0,
        signature=true_relation.signature,
        relation_tables=((),),
    )
    invalid_first = _typed_map(true_relation, false_relation, ())
    valid_second = _typed_map(false_relation, false_relation, ())

    with pytest.raises(OperationDomainValidationError) as error:
        compose_homomorphisms(invalid_first, valid_second)
    assert error.value.errors()[0]["type"] == (
        "relational.homomorphism.claim_not_preserved"
    )


def test_composition_rechecks_forged_json_homomorphism_claim() -> None:
    true_relation = FiniteRelationalStructure(
        carrier_size=0,
        signature=(FiniteRelationSymbol(symbol_id="T", arity=0),),
        relation_tables=(((),),),
    )
    false_relation = FiniteRelationalStructure(
        carrier_size=0,
        signature=true_relation.signature,
        relation_tables=((),),
    )
    decoded_claim = RelationalHomomorphism.model_validate_json(
        _typed_map(true_relation, false_relation, ()).model_dump_json()
    )
    valid_second = _typed_map(false_relation, false_relation, ())

    with pytest.raises(OperationDomainValidationError) as error:
        compose_homomorphisms(decoded_claim, valid_second)
    assert error.value.errors()[0]["type"] == (
        "relational.homomorphism.claim_not_preserved"
    )


def test_composition_rejects_model_constructed_malformed_map_shape() -> None:
    source = _structure(())
    malformed = RelationalHomomorphism.model_construct(
        source=source,
        target=source,
        mapping=(0,),
    )
    identity = _typed_map(source, source, (0, 1))

    with pytest.raises(OperationDomainValidationError) as error:
        compose_homomorphisms(malformed, identity)
    assert error.value.errors()[0]["type"] == ("relational.homomorphism.claim_shape")


def test_composition_requires_exact_intermediate_structure() -> None:
    source = _structure(())
    target_with_loop = _structure(((0, 0),))
    target_without_loop = _structure(())
    final = _structure(())

    first = _typed_map(source, target_with_loop, (0, 0))
    second = _typed_map(target_without_loop, final, (0, 1))
    with pytest.raises(OperationDomainValidationError) as error:
        compose_homomorphisms(first, second)
    assert error.value.errors()[0]["type"] == (
        "relational.homomorphism.intermediate_mismatch"
    )


def test_composition_admits_both_relation_replays_before_checking_claims() -> None:
    symbols = tuple(
        FiniteRelationSymbol(symbol_id=f"R{index}", arity=2) for index in range(5)
    )
    full_pairs = tuple(product(range(64), repeat=2))
    structure = FiniteRelationalStructure(
        carrier_size=64,
        signature=symbols,
        relation_tables=(full_pairs, full_pairs, full_pairs, full_pairs, ((0, 0),)),
    )
    identity = _typed_map(structure, structure, tuple(range(64)))

    with pytest.raises(OperationResourceAdmissionError) as error:
        compose_homomorphisms(identity, identity)
    assert error.value.errors()[0]["type"] == "relational.homomorphism.transport_bound"
    assert MAX_HOMOMORPHISM_COMPOSITION_TUPLE_REPLAYS == 32_768


def test_public_identity_and_compose_examples_use_canonical_value_type() -> None:
    by_id = {tool.operation_id: tool for tool in TOOLS}
    identity = by_id["relational_homomorphism.identity.compute"]
    compose = by_id["relational_homomorphism.compose.compute"]
    assert identity.result_type is RelationalHomomorphism
    assert compose.result_type is RelationalHomomorphism
    identity_request = identity.request_type.model_validate(identity.examples[0].input)
    compose_request = compose.request_type.model_validate(compose.examples[0].input)
    assert isinstance(identity.run(identity_request), RelationalHomomorphism)
    assert isinstance(compose.run(compose_request), RelationalHomomorphism)


def test_carrier_map_derives_injectivity_surjectivity_and_image() -> None:
    source = _structure(())
    target = FiniteRelationalStructure(
        carrier_size=3,
        signature=source.signature,
        relation_tables=((),),
    )
    mapping = RelationalCarrierMap(source=source, target=target, mapping=(2, 2))
    assert not mapping.injective
    assert not mapping.surjective
    assert mapping.image == (2,)
