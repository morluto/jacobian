from __future__ import annotations

import pytest

from jacobian.catalog.models import MathTool
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    FilteredChainMapPageZeroResult,
    FilteredChainMapRequest,
    filtered_chain_map_page_zero,
)
from jacobian.math.topology.chain_complexes.filtered_extensions_tools import TOOLS
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)


def _complex() -> ChainComplexValue:
    return ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(2,),
        differential_matrices=(),
    )


def _filtration(first: tuple[int, int]) -> tuple[FiltrationLevel, ...]:
    return (
        FiltrationLevel(
            subspaces=(FilteredSubspace(vectors=(first,)),),
        ),
        FiltrationLevel(
            subspaces=(FilteredSubspace(vectors=((1, 0), (0, 1))),),
        ),
    )


def test_page_zero_map_is_exact_in_nonmatching_quotient_bases() -> None:
    # The ambient map swaps coordinates. The source F_0 is <e_1>, while the
    # target F_0 is <e_2>; each associated-graded map is therefore the 1x1
    # identity although the ambient matrix is not the identity.
    request = FilteredChainMapRequest(
        source=_complex(),
        source_filtration=_filtration((1, 0)),
        target=_complex(),
        target_filtration=_filtration((0, 1)),
        maps=(((0, 1), (1, 0)),),
    )
    result = filtered_chain_map_page_zero(request)

    assert result.source_dimensions == ((1,), (1,))
    assert result.target_dimensions == ((1,), (1,))
    assert result.maps == ((((1,),),), (((1,),),))
    restored = FilteredChainMapPageZeroResult.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result


def test_page_zero_map_rejects_a_non_filtration_preserving_map() -> None:
    request = FilteredChainMapRequest(
        source=_complex(),
        source_filtration=_filtration((1, 0)),
        target=_complex(),
        target_filtration=_filtration((0, 1)),
        maps=(((1, 0), (0, 1)),),
    )
    with pytest.raises(ValueError, match="filtration preservation"):
        filtered_chain_map_page_zero(request)


def test_page_zero_map_commutes_with_the_graded_boundary() -> None:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 1),
        differential_matrices=(((1,),),),
    )
    filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=((1,),)),
            ),
        ),
    )
    result = filtered_chain_map_page_zero(
        FilteredChainMapRequest(
            source=complex_value,
            source_filtration=filtration,
            target=complex_value,
            target_filtration=filtration,
            maps=(((2,),), ((2,),)),
        )
    )
    assert result.maps == ((((2,),), ((2,),)),)


def test_page_zero_map_tool_has_the_native_result_contract() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "homological.filtered_chain_map.page_zero.compute"
    )
    assert isinstance(tool, MathTool)
    assert tool.result_type is FilteredChainMapPageZeroResult
