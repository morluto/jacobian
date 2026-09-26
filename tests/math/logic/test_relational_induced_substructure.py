"""Exact contract tests for source-bound induced substructures."""

from __future__ import annotations

from jacobian.math.logic.relational_structures import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
    HomomorphismStatus,
    InducedSubstructureResult,
    check_homomorphism,
    induced_substructure,
)
from jacobian.math.logic.relational_structures._admission import (
    admit_induced_substructure,
)


def _cycle() -> FiniteRelationalStructure:
    return FiniteRelationalStructure(
        carrier_size=3,
        signature=(FiniteRelationSymbol(symbol_id="E", arity=2),),
        relation_tables=(((0, 1), (1, 2), (2, 0)),),
    )


def test_ordered_induced_carrier_transports_relations_and_composes() -> None:
    source = _cycle()
    result = induced_substructure(source, (2, 0))

    assert result.source == source
    assert result.inclusion == (2, 0)
    assert result.substructure.carrier_size == 2
    assert result.substructure.signature == source.signature
    assert result.substructure.relation_tables == (((0, 1),),)

    inclusion = check_homomorphism(result.substructure, result.source, result.inclusion)
    assert inclusion.status is HomomorphismStatus.HOMOMORPHISM


def test_result_serialization_preserves_axes_and_relations() -> None:
    result = induced_substructure(_cycle(), (2, 0))
    restored = InducedSubstructureResult.model_validate_json(
        result.model_dump_json(), strict=True
    )
    assert restored == result
    assert restored.inclusion == (2, 0)
    assert restored.substructure.relation_tables == (((0, 1),),)


def test_empty_induced_carrier_retains_nullary_truth_values() -> None:
    source = FiniteRelationalStructure(
        carrier_size=2,
        signature=(
            FiniteRelationSymbol(symbol_id="E", arity=2),
            FiniteRelationSymbol(symbol_id="True0", arity=0),
            FiniteRelationSymbol(symbol_id="False0", arity=0),
        ),
        relation_tables=(((0, 1),), ((),), ()),
    )

    result = induced_substructure(source, ())
    assert result.inclusion == ()
    assert result.substructure.carrier_size == 0
    assert result.substructure.relation_tables == ((), ((),), ())


def test_full_substructure_can_use_and_preserve_identity_axis() -> None:
    source = _cycle()
    result = induced_substructure(source, (0, 1, 2))
    assert result.inclusion == (0, 1, 2)
    assert result.substructure == source


def test_admission_bounds_row_and_coordinate_work() -> None:
    source = _cycle()
    work = admit_induced_substructure(source, (2, 0))
    assert work == 9  # three binary rows, each charged for its row and coordinates
