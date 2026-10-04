from __future__ import annotations

from typing import cast

import pytest
from tests.error_assertions import error_code

from jacobian.catalog.models import MathTool, OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes import filtered_extensions
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    FilteredChainMapCompositionRequest,
    FilteredChainMapRequest,
    FilteredChainMapResult,
    filtered_chain_map_compose,
    filtered_map,
)
from jacobian.math.topology.chain_complexes.filtered_extensions_tools import TOOLS
from jacobian.math.topology.chain_complexes.operations import chain_map_commutes
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    ChainMapValue,
    CoefficientRing,
)


def _complex() -> ChainComplexValue:
    return ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 1),
        differential_matrices=(((1,),),),
    )


def _filtration() -> tuple[FiltrationLevel, ...]:
    # A one-level filtration is the degenerate case F_0 C = C.
    return (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=((1,),)),
            ),
        ),
    )


def _map(scalar: int) -> FilteredChainMapRequest:
    complex_value = _complex()
    filtration = _filtration()
    return FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=complex_value,
            target=complex_value,
            map_matrices=(((scalar,),), ((scalar,),)),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
    )


def _composition(
    first: FilteredChainMapRequest, second: FilteredChainMapRequest
) -> tuple[FilteredChainMapResult, FilteredChainMapResult]:
    return filtered_map(first), filtered_map(second)


def _native_compose(
    request: FilteredChainMapCompositionRequest,
) -> FilteredChainMapResult:
    return filtered_chain_map_compose(request.first, request.second)


def test_composition_is_exact_and_remains_a_chain_map_after_roundtrip() -> None:
    result = filtered_chain_map_compose(*_composition(_map(2), _map(3)))

    assert result.chain_map.map_matrices == (((6,),), ((6,),))
    assert result.filtration_preserving and result.chain_map
    from jacobian.math.topology.chain_complexes.values import ChainMapValue

    assert chain_map_commutes(
        ChainMapValue(
            source=result.chain_map.source,
            target=result.chain_map.target,
            map_matrices=result.chain_map.map_matrices,
        )
    ).is_valid
    # The output's source, target, filtration and matrices can be consumed
    # unchanged as the next filtered-map input after JSON serialization.
    restored = FilteredChainMapResult.model_validate_json(result.model_dump_json())
    continued = filtered_chain_map_compose(restored, filtered_map(_map(1)))
    assert continued.chain_map.map_matrices == result.chain_map.map_matrices
    replayed = filtered_map(
        FilteredChainMapRequest(
            chain_map=ChainMapValue(
                source=restored.chain_map.source,
                target=restored.chain_map.target,
                map_matrices=restored.chain_map.map_matrices,
            ),
            source_filtration=restored.source_filtration,
            target_filtration=restored.target_filtration,
        )
    )
    assert replayed.chain_map and replayed.filtration_preserving


def test_composition_rejects_a_mismatched_middle_filtration() -> None:
    two_dimensional = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(2,),
        differential_matrices=(),
    )
    full = FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0), (0, 1))),))
    first = FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=two_dimensional,
            target=two_dimensional,
            map_matrices=(((2, 0), (0, 2)),),
        ),
        source_filtration=(
            FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),)),
            full,
        ),
        target_filtration=(
            FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),)),
            full,
        ),
    )
    second = FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=two_dimensional,
            target=two_dimensional,
            map_matrices=(((3, 0), (0, 3)),),
        ),
        source_filtration=(
            FiltrationLevel(subspaces=(FilteredSubspace(vectors=((0, 1),)),)),
            full,
        ),
        target_filtration=(
            FiltrationLevel(subspaces=(FilteredSubspace(vectors=((0, 1),)),)),
            full,
        ),
    )
    with pytest.raises(ValueError) as exc_info:
        filtered_chain_map_compose(*_composition(first, second))
    assert (
        error_code(exc_info.value)
        == "filtered_chain_map.composition_middle_filtration_mismatch"
    )


def test_middle_filtrations_may_use_different_spanning_vectors() -> None:
    two_dimensional = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(2,),
        differential_matrices=(),
    )
    first_level = FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),))
    equivalent_level = FiltrationLevel(
        subspaces=(FilteredSubspace(vectors=((1, 0), (2, 0))),)
    )
    full = FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0), (0, 1))),))
    first = FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=two_dimensional,
            target=two_dimensional,
            map_matrices=(((2, 0), (0, 2)),),
        ),
        source_filtration=(first_level, full),
        target_filtration=(first_level, full),
    )
    second = FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=two_dimensional,
            target=two_dimensional,
            map_matrices=(((3, 0), (0, 3)),),
        ),
        source_filtration=(equivalent_level, full),
        target_filtration=(first_level, full),
    )

    result = filtered_chain_map_compose(*_composition(first, second))
    assert result.chain_map.map_matrices == (((6, 0), (0, 6)),)


def test_composition_rejects_a_caller_supplied_non_chain_map() -> None:
    base = _map(1)
    invalid = base.model_copy(
        update={
            "chain_map": base.chain_map.model_copy(
                update={"map_matrices": (((1,),), ((2,),))}
            )
        }
    )
    forged_profile = filtered_map(invalid).model_copy(update={"chain_map": True})
    with pytest.raises(ValueError, match="carry canonical chain maps"):
        filtered_chain_map_compose(forged_profile, filtered_map(_map(3)))


def test_identity_composition_accepts_maximum_bounded_coefficient() -> None:
    large_scalar = 10**4095
    result = _native_compose(
        FilteredChainMapCompositionRequest(
            first=filtered_map(_map(1)),
            second=filtered_map(_map(large_scalar)),
        )
    )
    assert result.chain_map.map_matrices == (((large_scalar,),), ((large_scalar,),))


def test_composition_admits_exact_coefficient_growth_before_multiplication() -> None:
    large_scalar = 10**3000
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        _native_compose(
            FilteredChainMapCompositionRequest(
                first=filtered_map(_map(large_scalar)),
                second=filtered_map(_map(large_scalar)),
            )
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "filtered_chain_map.composition_coefficient_exceeded"
    )


def test_composition_operation_exposes_the_native_contract() -> None:
    tool = cast(
        MathTool[FilteredChainMapCompositionRequest, FilteredChainMapResult],
        next(
            item
            for item in TOOLS
            if item.operation_id == "homological.filtered_chain_map.compose.compute"
        ),
    )
    assert tool.result_type is FilteredChainMapResult
    first, second = _composition(_map(2), _map(3))
    request = FilteredChainMapCompositionRequest(first=first, second=second)
    assert tool.run(request).chain_map.map_matrices == (((6,),), ((6,),))


def test_composition_rejects_malformed_component_axes_before_indexing() -> None:
    first, second = _composition(_map(1), _map(2))
    malformed = FilteredChainMapCompositionRequest(
        first=first, second=second
    ).model_copy(
        update={
            "first": first.model_copy(
                update={
                    "chain_map": first.chain_map.model_copy(
                        update={
                            "target": ChainComplexValue(
                                coefficient_ring=CoefficientRing.RATIONAL,
                                degree_min=0,
                                degree_max=0,
                                basis_sizes=(1,),
                                differential_matrices=(),
                            )
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError) as exc_info:
        _native_compose(malformed)
    assert (
        error_code(exc_info.value) == "filtered_chain_map.composition_middle_mismatch"
    )


def test_composition_rejects_non_result_components_at_native_boundary() -> None:
    malformed = FilteredChainMapCompositionRequest.model_construct(
        first={"source": None}, second=filtered_map(_map(1))
    )
    with pytest.raises(ValueError, match="both composition components"):
        _native_compose(malformed)


def test_composition_counts_integer_input_characters_without_fraction_overhead() -> (
    None
):
    from fractions import Fraction

    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(4,),
        differential_matrices=(),
    )
    identity = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    filtration = (FiltrationLevel(subspaces=(FilteredSubspace(vectors=identity),)),)
    unit = FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=complex_value,
            target=complex_value,
            map_matrices=(identity,),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
    )
    large = 10**4095
    large_map = unit.model_copy(
        update={
            "chain_map": ChainMapValue(
                source=complex_value,
                target=complex_value,
                map_matrices=(
                    tuple(
                        tuple(large if row == column else 0 for column in range(4))
                        for row in range(4)
                    ),
                ),
            )
        }
    )
    result = filtered_chain_map_compose(*_composition(unit, large_map))
    assert result.chain_map.map_matrices == (
        tuple(
            tuple(Fraction(value) for value in row)
            for row in large_map.chain_map.map_matrices[0]
        ),
    )


def test_composition_preflights_semantic_work_before_exact_admission() -> None:
    large = 10**3000
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(32,),
        differential_matrices=(),
    )
    basis = tuple(tuple(large if i == j else 0 for j in range(32)) for i in range(32))
    filtration = (
        FiltrationLevel(
            subspaces=(FilteredSubspace(vectors=basis + basis),),
        ),
    )
    identity = tuple(tuple(1 if i == j else 0 for j in range(32)) for i in range(32))
    component = FilteredChainMapResult(
        chain_map=ChainMapValue(
            source=complex_value,
            target=complex_value,
            map_matrices=(identity,),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
        filtration_preserving=True,
        is_chain_map=True,
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        _native_compose(
            FilteredChainMapCompositionRequest(first=component, second=component)
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "filtered_chain_map.composition_work_exceeded"
    )


def test_composition_rejects_oversized_map_shape_before_parsing_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=4,
        basis_sizes=(32,) * 5,
        differential_matrices=tuple(
            tuple((0,) * 32 for _ in range(32)) for _ in range(4)
        ),
    )
    identity = tuple(tuple(1 if i == j else 0 for j in range(32)) for i in range(32))
    filtration = (
        FiltrationLevel(
            subspaces=tuple(
                FilteredSubspace(
                    vectors=tuple(
                        tuple(1 if i == j else 0 for j in range(32)) for i in range(32)
                    )
                )
                for _ in range(5)
            ),
        ),
    )
    component = FilteredChainMapResult.model_construct(
        chain_map=ChainMapValue.model_construct(
            source=complex_value,
            target=complex_value,
            map_matrices=(identity,) * 5,
        ),
        source_filtration=filtration,
        target_filtration=filtration,
        filtration_preserving=True,
        is_chain_map=True,
    )

    def parsing_was_not_reached(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "map coefficients must not be parsed before shape admission"
        )

    monkeypatch.setattr(filtered_extensions, "_parse_entry", parsing_was_not_reached)
    with pytest.raises(OperationResourceAdmissionError, match="cell envelope"):
        _native_compose(
            FilteredChainMapCompositionRequest(first=component, second=component)
        )


def test_composition_rejects_rational_coefficient_in_finite_field_map() -> None:
    from fractions import Fraction

    finite = ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=3,
        degree_min=0,
        degree_max=0,
        basis_sizes=(1,),
        differential_matrices=(),
    )
    filtration = (FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1,),)),)),)
    forged = FilteredChainMapResult.model_construct(
        chain_map=ChainMapValue.model_construct(
            source=finite,
            target=finite,
            map_matrices=(((Fraction(1, 2),),),),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
        filtration_preserving=True,
        is_chain_map=True,
    )
    request = FilteredChainMapCompositionRequest(
        first=forged,
        second=filtered_map(
            FilteredChainMapRequest(
                chain_map=ChainMapValue(
                    source=finite,
                    target=finite,
                    map_matrices=(((1,),),),
                ),
                source_filtration=filtration,
                target_filtration=filtration,
            )
        ),
    )
    with pytest.raises(ValueError) as exc_info:
        _native_compose(request)
    assert error_code(exc_info.value) == "filtered_chain_map.entry_invalid"


@pytest.mark.parametrize("entry", [3, -1])
def test_filtered_map_rejects_noncanonical_finite_field_residues(entry: int) -> None:
    finite = ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=3,
        degree_min=0,
        degree_max=0,
        basis_sizes=(1,),
        differential_matrices=(),
    )
    filtration = (FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1,),)),)),)
    # Main validates the coefficient grammar both at ChainMapValue construction
    # and on re-admission inside the operation, so a forged non-residue is
    # rejected with the model-layer residue error before any grammar parsing.
    from pydantic import ValidationError as _ValidationError

    with pytest.raises(_ValidationError) as exc_info:
        filtered_map(
            FilteredChainMapRequest.model_construct(
                chain_map=ChainMapValue.model_construct(
                    source=finite,
                    target=finite,
                    map_matrices=(((entry,),),),
                ),
                source_filtration=filtration,
                target_filtration=filtration,
            )
        )
    assert (
        exc_info.value.errors()[0]["type"]
        == "chain_complex.prime_field_residue_invalid"
    )


def test_composition_output_admission_counts_only_integer_digits() -> None:
    dimension = 4
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(dimension,),
        differential_matrices=(),
    )
    filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(
                    vectors=tuple(
                        tuple(int(i == j) for i in range(dimension))
                        for j in range(dimension)
                    )
                ),
            ),
        ),
    )
    scalar = 10**4095
    identity = FilteredChainMapResult(
        chain_map=ChainMapValue(
            source=complex_value,
            target=complex_value,
            map_matrices=(
                tuple(
                    tuple(int(i == j) for j in range(dimension))
                    for i in range(dimension)
                ),
            ),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
        filtration_preserving=True,
        is_chain_map=True,
    )
    dense = identity.model_copy(
        update={
            "chain_map": identity.chain_map.model_copy(
                update={
                    "map_matrices": (
                        tuple(
                            tuple(scalar for _ in range(dimension))
                            for _ in range(dimension)
                        ),
                    )
                }
            )
        }
    )
    assert (
        filtered_chain_map_compose(identity, dense).chain_map.map_matrices[0][0][0]
        == scalar
    )


def test_composition_input_admission_counts_negative_signs() -> None:
    dimension = 4
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(dimension,),
        differential_matrices=(),
    )
    filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(
                    vectors=tuple(
                        tuple(int(i == j) for i in range(dimension))
                        for j in range(dimension)
                    )
                ),
            ),
        ),
    )
    scalar = -(10**4095)
    # Forge past ChainMapValue's own entry budget so the operation's digit
    # admission is what counts the negative sign.
    dense = FilteredChainMapResult.model_construct(
        chain_map=ChainMapValue.model_construct(
            source=complex_value,
            target=complex_value,
            map_matrices=(
                tuple(
                    tuple(scalar for _ in range(dimension)) for _ in range(dimension)
                ),
            ),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
        filtration_preserving=True,
        is_chain_map=True,
    )
    identity = dense.model_copy(
        update={
            "chain_map": dense.chain_map.model_copy(
                update={
                    "map_matrices": (
                        tuple(
                            tuple(int(i == j) for j in range(dimension))
                            for i in range(dimension)
                        ),
                    )
                }
            )
        }
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="cell or coefficient-character envelope"
    ):
        filtered_chain_map_compose(identity, dense)
