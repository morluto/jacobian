from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.dispatch import invoke_operation
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    FilteredChainMapCompositionRequest,
    FilteredChainMapPageRequest,
    FilteredChainMapPageResult,
    FilteredChainMapRequest,
    filtered_chain_map_compose,
    filtered_chain_map_page,
    filtered_map,
)
from jacobian.math.topology.chain_complexes.filtered_extensions_tools import TOOLS
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)


def _complex() -> ChainComplexValue:
    # d(a)=x lowers filtration by exactly one, so d_1 is the identity
    # between E_1^(1,0) and E_1^(0,0).
    return ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 1),
        differential_matrices=(((1,),),),
    )


def _filtration() -> tuple[FiltrationLevel, ...]:
    return (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=()),
            ),
        ),
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=((1,),)),
            ),
        ),
    )


def _map(scale: int):
    complex_value = _complex()
    filtration = _filtration()
    return filtered_map(
        FilteredChainMapRequest(
            source=complex_value,
            source_filtration=filtration,
            target=complex_value,
            target_filtration=filtration,
            maps=(((scale,),), ((scale,),)),
        )
    )


def test_e1_page_map_transports_representatives_and_commutes_with_d1() -> None:
    value = filtered_chain_map_page(FilteredChainMapPageRequest(map=_map(1), page=1))

    # Independent tiny oracle: E0 has two one-dimensional terms and zero d0;
    # its homology is the same two terms, and the representative d1 is d(a)=x.
    assert value.source_page.page_dimensions == ((1, 0), (0, 1))
    assert value.source_page.differentials[0].entries == ((1,),)
    assert value.maps == ((((1,),), ()), ((), ((1,),)))
    assert value.source_page == value.target_page
    assert (
        FilteredChainMapPageResult.model_validate_json(value.model_dump_json()) == value
    )


def test_page_maps_preserve_composition_on_a_late_page() -> None:
    first, second = _map(2), _map(3)
    composite = filtered_chain_map_compose(
        # Arguments are in application order.
        FilteredChainMapCompositionRequest(first=first, second=second)
    )
    direct = filtered_chain_map_page(FilteredChainMapPageRequest(map=composite, page=1))
    first_page = filtered_chain_map_page(FilteredChainMapPageRequest(map=first, page=1))
    second_page = filtered_chain_map_page(
        FilteredChainMapPageRequest(map=second, page=1)
    )
    assert first_page.maps == ((((2,),), ()), ((), ((2,),)))
    assert second_page.maps == ((((3,),), ()), ((), ((3,),)))
    assert direct.maps == ((((6,),), ()), ((), ((6,),)))


def test_page_two_map_handles_a_differential_after_zero_d1() -> None:
    complex_value = _complex()
    low = FiltrationLevel(
        subspaces=(
            FilteredSubspace(vectors=((1,),)),
            FilteredSubspace(vectors=()),
        )
    )
    filtration = (
        low,
        low,
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=((1,),)),
            )
        ),
    )
    chain_map = filtered_map(
        FilteredChainMapRequest(
            source=complex_value,
            source_filtration=filtration,
            target=complex_value,
            target_filtration=filtration,
            maps=(((2,),), ((2,),)),
        )
    )

    # E0 has a class x in level 0 and a in level 2. There is no level-1
    # target, so d1(a)=0; the chain differential d(a)=x gives d2(a)=x.
    page_one = filtered_chain_map_page(
        FilteredChainMapPageRequest(map=chain_map, page=1)
    )
    assert all(
        not any(any(entry for entry in row) for row in record.entries)
        for record in page_one.source_page.differentials
    )
    page_two = filtered_chain_map_page(
        FilteredChainMapPageRequest(map=chain_map, page=2)
    )
    d2 = next(
        record
        for record in page_two.source_page.differentials
        if record.source_level == 2 and record.source_degree == 1
    )
    assert d2.entries == ((1,),)
    assert page_two.maps == ((((2,),), ()), ((), ()), ((), ((2,),)))


def test_zero_page_map_preserves_empty_page_axes() -> None:
    empty = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(0,),
        differential_matrices=(),
    )
    filtration = (FiltrationLevel(subspaces=(FilteredSubspace(vectors=()),)),)
    map_value = filtered_map(
        FilteredChainMapRequest(
            source=empty,
            source_filtration=filtration,
            target=empty,
            target_filtration=filtration,
            maps=((),),
        )
    )
    page_map = filtered_chain_map_page(
        FilteredChainMapPageRequest(map=map_value, page=MAX_PAGE)
    )
    assert page_map.maps == (((),),)
    assert page_map.source_page.page_dimensions == ((0,),)


MAX_PAGE = 4


def test_page_map_rejects_a_false_chain_map_claim() -> None:
    authored = _map(1).model_copy(update={"maps": (((1,),), ((2,),))})
    with pytest.raises(ValueError, match="chain map"):
        filtered_chain_map_page(FilteredChainMapPageRequest(map=authored, page=1))


def test_page_map_tool_uses_the_native_operation_contract() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "homological.filtered_chain_map.page.compute"
    )
    assert isinstance(tool, MathTool)
    assert tool.result_type is FilteredChainMapPageResult
    example = tool.examples[0]
    result = invoke_operation(
        tool.operation_id,
        example.input,
        Catalog.open(),
    )
    assert result.output["maps"] == [[[["1"]]]]
