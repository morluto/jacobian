from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes import filtered_extensions
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


def test_page_zero_map_retains_its_quotient_axes() -> None:
    result = filtered_chain_map_page_zero(
        FilteredChainMapRequest(
            source=_complex(),
            source_filtration=_filtration((1, 0)),
            target=_complex(),
            target_filtration=_filtration((0, 1)),
            maps=(((0, 1), (1, 0)),),
        )
    )

    assert result.source_representatives == ((((1, 0),),), (((0, 1),),))
    assert result.target_representatives == ((((0, 1),),), (((1, 0),),))
    # The ambient swap is the identity once expressed in these explicit axes.
    assert result.maps == ((((1,),),), (((1,),),))
    assert (
        FilteredChainMapPageZeroResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_page_zero_map_rejects_negative_graded_dimensions() -> None:
    baseline = filtered_chain_map_page_zero(
        FilteredChainMapRequest(
            source=_complex(),
            source_filtration=_filtration((1, 0)),
            target=_complex(),
            target_filtration=_filtration((0, 1)),
            maps=(((0, 1), (1, 0)),),
        )
    )
    data = baseline.model_dump()
    # A negative dimension offset by an oversized later dimension still sums to
    # the ambient rank, so only the per-dimension bound rejects it.
    data["source_dimensions"] = ((-1,), (3,))
    with pytest.raises(ValueError, match="between zero and each chain rank"):
        FilteredChainMapPageZeroResult.model_validate(data)


def test_page_zero_map_translates_a_finite_field_rational_entry() -> None:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=2,
        degree_min=0,
        degree_max=0,
        basis_sizes=(1,),
        differential_matrices=(),
    )
    filtration = (FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1,),)),)),)
    request = FilteredChainMapRequest(
        source=complex_value,
        source_filtration=filtration,
        target=complex_value,
        target_filtration=filtration,
        maps=(((Fraction(1, 2),),),),
    )
    with pytest.raises(
        OperationDomainValidationError, match="canonical coefficient grammar"
    ):
        filtered_chain_map_page_zero(request)


def _non_exhaustive_request() -> FilteredChainMapRequest:
    # The top level does not span e_2, so semantic admission would reject the
    # request; a resource preflight must fire before that exact expansion.
    filtration = (
        FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),)),
        FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),)),
    )
    return FilteredChainMapRequest(
        source=_complex(),
        source_filtration=filtration,
        target=_complex(),
        target_filtration=filtration,
        maps=(((1, 0), (0, 1)),),
    )


def test_page_zero_map_preflights_work_before_semantic_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(filtered_extensions, "MAX_FILTERED_HOMOLOGY_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        filtered_chain_map_page_zero(_non_exhaustive_request())
    assert excinfo.value.errors()[0]["type"] == "filtered_chain_map.e0_work_exceeded"


def test_page_zero_map_preflights_output_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(filtered_extensions, "MAX_FILTERED_HOMOLOGY_RESULT_CHARS", 0)
    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        filtered_chain_map_page_zero(_non_exhaustive_request())
    assert (
        excinfo.value.errors()[0]["type"]
        == "filtered_chain_map.e0_output_chars_exceeded"
    )


def test_page_zero_map_tool_has_the_native_result_contract() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "homological.filtered_chain_map.page_zero.compute"
    )
    assert isinstance(tool, MathTool)
    assert tool.result_type is FilteredChainMapPageZeroResult
