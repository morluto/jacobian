from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes import filtered_extensions
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    FilteredChainMapPageRequest,
    FilteredChainMapPageResult,
    FilteredChainMapRequest,
    FilteredChainMapResult,
    filtered_chain_map_compose,
    filtered_chain_map_page,
    filtered_map,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    ChainMapValue,
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
            chain_map=ChainMapValue(
                source=complex_value,
                target=complex_value,
                map_matrices=(((scale,),), ((scale,),)),
            ),
            source_filtration=filtration,
            target_filtration=filtration,
        )
    )


def test_page_map_prices_scalar_height_and_vector_count_before_elimination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = 32
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(rank,),
        differential_matrices=(),
    )
    large = 10**4095
    filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(
                    vectors=tuple(
                        tuple(large if row == column else 0 for column in range(rank))
                        for row in range(rank)
                    )
                ),
            )
        ),
    )
    identity = tuple(
        tuple(1 if row == column else 0 for column in range(rank))
        for row in range(rank)
    )
    authored = FilteredChainMapResult.model_construct(
        chain_map=ChainMapValue.model_construct(
            source=complex_value,
            target=complex_value,
            map_matrices=(identity,),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
        filtration_preserving=True,
        is_chain_map=True,
    )
    request = FilteredChainMapPageRequest(map=authored, page=1)
    monkeypatch.setattr(
        filtered_extensions,
        "_admit_filtered_semantics",
        lambda *_args: pytest.fail(
            "exact filtration elimination began before admission"
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        filtered_chain_map_page(request.map, request.page)

    assert error.value.errors()[0]["type"] == "filtered_chain_map.page_work_exceeded"


def test_e1_page_map_transports_representatives_and_commutes_with_d1() -> None:
    value = filtered_chain_map_page(_map(1), 1)

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
    composite = filtered_chain_map_compose(first, second)
    direct = filtered_chain_map_page(composite, 1)
    first_page = filtered_chain_map_page(first, 1)
    second_page = filtered_chain_map_page(second, 1)
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
            chain_map=ChainMapValue(
                source=complex_value,
                target=complex_value,
                map_matrices=(((2,),), ((2,),)),
            ),
            source_filtration=filtration,
            target_filtration=filtration,
        )
    )

    # E0 has a class x in level 0 and a in level 2. There is no level-1
    # target, so d1(a)=0; the chain differential d(a)=x gives d2(a)=x.
    page_one = filtered_chain_map_page(chain_map, 1)
    assert all(
        not any(any(entry for entry in row) for row in record.entries)
        for record in page_one.source_page.differentials
    )
    page_two = filtered_chain_map_page(chain_map, 2)
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
            chain_map=ChainMapValue(
                source=empty,
                target=empty,
                map_matrices=((),),
            ),
            source_filtration=filtration,
            target_filtration=filtration,
        )
    )
    page_map = filtered_chain_map_page(map_value, MAX_PAGE)
    assert page_map.maps == (((),),)
    assert page_map.source_page.page_dimensions == ((0,),)


MAX_PAGE = 4


def test_page_map_rejects_malformed_native_page_before_arithmetic() -> None:
    request = FilteredChainMapPageRequest(map=_map(1), page=1).model_copy(
        update={"page": "1"}
    )
    with pytest.raises(ValueError) as exc_info:
        filtered_chain_map_page(request.map, request.page)
    assert exc_info.value.errors()[0]["type"] == "filtered_chain_map.page_invalid"


def test_page_map_rejects_a_false_chain_map_claim() -> None:
    base = _map(1)
    authored = base.model_copy(
        update={
            "chain_map": base.chain_map.model_copy(
                update={"map_matrices": (((1,),), ((2,),))}
            )
        }
    )
    with pytest.raises(ValueError) as exc_info:
        filtered_chain_map_page(authored, 1)
    assert exc_info.value.errors()[0]["type"] == "filtered_chain_map.not_chain_map"


def test_page_map_admits_sparse_large_coefficients_by_entry() -> None:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(3,),
        differential_matrices=(),
    )
    filtration = (
        FiltrationLevel(
            subspaces=(FilteredSubspace(vectors=((1, 0, 0), (0, 1, 0), (0, 0, 1))),)
        ),
    )
    large = 10**4095
    chain_map = filtered_map(
        FilteredChainMapRequest(
            chain_map=ChainMapValue(
                source=complex_value,
                target=complex_value,
                map_matrices=(((large, 0, 0), (0, 1, 0), (0, 0, 1)),),
            ),
            source_filtration=filtration,
            target_filtration=filtration,
        )
    )

    result = filtered_chain_map_page(chain_map, 1)

    assert result.maps == ((((large, 0, 0), (0, 1, 0), (0, 0, 1)),),)


@pytest.mark.parametrize("nested_map", [{"source": None}, None])
def test_page_map_rejects_forged_nested_map_at_native_boundary(nested_map) -> None:
    request = FilteredChainMapPageRequest.model_construct(map=nested_map, page=1)
    with pytest.raises(OperationDomainValidationError) as exc_info:
        filtered_chain_map_page(request.map, request.page)
    assert exc_info.value.errors()[0]["type"] == "filtered_chain_map.page_map_invalid"
