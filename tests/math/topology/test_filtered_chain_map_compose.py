from __future__ import annotations

import pytest

from jacobian.catalog.models import MathTool, OperationResourceAdmissionError
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
        source=complex_value,
        source_filtration=filtration,
        target=complex_value,
        target_filtration=filtration,
        maps=(((scalar,),), ((scalar,),)),
    )


def _composition(
    first: FilteredChainMapRequest, second: FilteredChainMapRequest
) -> FilteredChainMapCompositionRequest:
    return FilteredChainMapCompositionRequest(
        first=filtered_map(first), second=filtered_map(second)
    )


def test_composition_is_exact_and_remains_a_chain_map_after_roundtrip() -> None:
    result = filtered_chain_map_compose(_composition(_map(2), _map(3)))

    assert result.maps == (((6,),), ((6,),))
    assert result.filtration_preserving and result.chain_map
    assert chain_map_commutes(result.source, result.target, result.maps).is_valid
    # The output's source, target, filtration and matrices can be consumed
    # unchanged as the next filtered-map input after JSON serialization.
    restored = FilteredChainMapResult.model_validate_json(result.model_dump_json())
    continued = filtered_chain_map_compose(
        FilteredChainMapCompositionRequest(
            first=restored,
            second=filtered_map(_map(1)),
        )
    )
    assert continued.maps == result.maps
    replayed = filtered_map(
        FilteredChainMapRequest(
            source=restored.source,
            source_filtration=restored.source_filtration,
            target=restored.target,
            target_filtration=restored.target_filtration,
            maps=restored.maps,
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
        source=two_dimensional,
        source_filtration=(
            FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),)),
            full,
        ),
        target=two_dimensional,
        target_filtration=(
            FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),)),
            full,
        ),
        maps=(((2, 0), (0, 2)),),
    )
    second = FilteredChainMapRequest(
        source=two_dimensional,
        source_filtration=(
            FiltrationLevel(subspaces=(FilteredSubspace(vectors=((0, 1),)),)),
            full,
        ),
        target=two_dimensional,
        target_filtration=(
            FiltrationLevel(subspaces=(FilteredSubspace(vectors=((0, 1),)),)),
            full,
        ),
        maps=(((3, 0), (0, 3)),),
    )
    with pytest.raises(ValueError, match="middle filtrations"):
        filtered_chain_map_compose(_composition(first, second))


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
        source=two_dimensional,
        source_filtration=(first_level, full),
        target=two_dimensional,
        target_filtration=(first_level, full),
        maps=(((2, 0), (0, 2)),),
    )
    second = FilteredChainMapRequest(
        source=two_dimensional,
        source_filtration=(equivalent_level, full),
        target=two_dimensional,
        target_filtration=(first_level, full),
        maps=(((3, 0), (0, 3)),),
    )

    result = filtered_chain_map_compose(_composition(first, second))
    assert result.maps == (((6, 0), (0, 6)),)


def test_composition_rejects_a_caller_supplied_non_chain_map() -> None:
    invalid = _map(1).model_copy(update={"maps": (((1,),), ((2,),))})
    forged_profile = filtered_map(invalid).model_copy(update={"chain_map": True})
    with pytest.raises(ValueError, match="must commute with their chain differentials"):
        filtered_chain_map_compose(
            FilteredChainMapCompositionRequest(
                first=forged_profile, second=filtered_map(_map(3))
            )
        )


def test_identity_composition_accepts_maximum_bounded_coefficient() -> None:
    large_scalar = 10**4095
    result = filtered_chain_map_compose(
        FilteredChainMapCompositionRequest(
            first=filtered_map(_map(1)),
            second=filtered_map(_map(large_scalar)),
        )
    )
    assert result.maps == (((large_scalar,),), ((large_scalar,),))


def test_composition_admits_exact_coefficient_growth_before_multiplication() -> None:
    large_scalar = 10**3000
    with pytest.raises(
        OperationResourceAdmissionError,
        match="composed coefficient may exceed",
    ):
        filtered_chain_map_compose(
            FilteredChainMapCompositionRequest(
                first=filtered_map(_map(large_scalar)),
                second=filtered_map(_map(large_scalar)),
            )
        )


def test_composition_operation_exposes_the_native_contract() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "homological.filtered_chain_map.compose.compute"
    )
    assert isinstance(tool, MathTool)
    assert tool.result_type is FilteredChainMapResult
    request = _composition(_map(2), _map(3))
    assert tool.run(request).maps == (((6,),), ((6,),))


def test_composition_rejects_malformed_component_axes_before_indexing() -> None:
    malformed = _composition(_map(1), _map(2)).model_copy(
        update={
            "first": _composition(_map(1), _map(2)).first.model_copy(
                update={"target": ChainComplexValue(coefficient_ring=CoefficientRing.RATIONAL, degree_min=0, degree_max=0, basis_sizes=(1,), differential_matrices=())}
            )
        }
    )
    with pytest.raises(ValueError, match="target complex of the first map"):
        filtered_chain_map_compose(malformed)


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
    forged = FilteredChainMapResult(
        source=finite, target=finite,
        source_filtration=filtration, target_filtration=filtration,
        maps=(((Fraction(1, 2),),),), filtration_preserving=True, chain_map=True,
    )
    request = FilteredChainMapCompositionRequest(first=forged, second=filtered_map(
        FilteredChainMapRequest(source=finite, target=finite, source_filtration=filtration,
                                target_filtration=filtration, maps=(((1,),),))))
    with pytest.raises(ValueError, match="finite-field map entries must be integers"):
        filtered_chain_map_compose(request)
