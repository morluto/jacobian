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
    ChainMapValue,
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
        chain_map=ChainMapValue(
            source=_complex(),
            target=_complex(),
            map_matrices=(((0, 1), (1, 0)),),
        ),
        source_filtration=_filtration((1, 0)),
        target_filtration=_filtration((0, 1)),
    )
    result = filtered_chain_map_page_zero(
        request.chain_map, request.source_filtration, request.target_filtration
    )

    assert result.source_dimensions == ((1,), (1,))
    assert result.target_dimensions == ((1,), (1,))
    assert result.maps == ((((1,),),), (((1,),),))
    restored = FilteredChainMapPageZeroResult.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result


def test_page_zero_map_rejects_a_non_filtration_preserving_map() -> None:
    request = FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=_complex(),
            target=_complex(),
            map_matrices=(((1, 0), (0, 1)),),
        ),
        source_filtration=_filtration((1, 0)),
        target_filtration=_filtration((0, 1)),
    )
    with pytest.raises(ValueError, match="filtration preservation"):
        filtered_chain_map_page_zero(
            request.chain_map, request.source_filtration, request.target_filtration
        )


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
        ChainMapValue(
            source=complex_value,
            target=complex_value,
            map_matrices=(((2,),), ((2,),)),
        ),
        filtration,
        filtration,
    )
    assert result.maps == ((((2,),), ((2,),)),)


def test_page_zero_map_retains_its_quotient_axes() -> None:
    result = filtered_chain_map_page_zero(
        ChainMapValue(
            source=_complex(),
            target=_complex(),
            map_matrices=(((0, 1), (1, 0)),),
        ),
        _filtration((1, 0)),
        _filtration((0, 1)),
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
        ChainMapValue(
            source=_complex(),
            target=_complex(),
            map_matrices=(((0, 1), (1, 0)),),
        ),
        _filtration((1, 0)),
        _filtration((0, 1)),
    )
    data = baseline.model_dump()
    # A negative dimension offset by an oversized later dimension still sums to
    # the ambient rank, so only the per-dimension bound rejects it.
    data["source_dimensions"] = ((-1,), (3,))
    with pytest.raises(ValueError, match="E0 quotient bases must match"):
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
    # Bypass ChainMapValue's own grammar check so the operation's re-admission
    # is what rejects the rational entry on a prime-field curve.
    request = FilteredChainMapRequest.model_construct(
        chain_map=ChainMapValue.model_construct(
            source=complex_value,
            target=complex_value,
            map_matrices=(((Fraction(1, 2),),),),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
    )
    with pytest.raises(
        OperationDomainValidationError, match="canonical coefficient grammar"
    ):
        filtered_chain_map_page_zero(
            request.chain_map, request.source_filtration, request.target_filtration
        )


def _non_exhaustive_request() -> FilteredChainMapRequest:
    # The top level does not span e_2, so semantic admission would reject the
    # request; a resource preflight must fire before that exact expansion.
    filtration = (
        FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),)),
        FiltrationLevel(subspaces=(FilteredSubspace(vectors=((1, 0),)),)),
    )
    return FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=_complex(),
            target=_complex(),
            map_matrices=(((1, 0), (0, 1)),),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
    )


def test_page_zero_map_preflights_work_before_semantic_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(filtered_extensions, "MAX_FILTERED_HOMOLOGY_WORK", 0)
    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        _r = _non_exhaustive_request()
        filtered_chain_map_page_zero(
            _r.chain_map, _r.source_filtration, _r.target_filtration
        )
    assert excinfo.value.errors()[0]["type"] == "filtered_chain_map.e0_work_exceeded"


def test_page_zero_map_preflights_output_digits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(filtered_extensions, "MAX_FILTERED_HOMOLOGY_RESULT_DIGITS", 0)
    with pytest.raises(OperationResourceAdmissionError) as excinfo:
        _r = _non_exhaustive_request()
        filtered_chain_map_page_zero(
            _r.chain_map, _r.source_filtration, _r.target_filtration
        )
    assert (
        excinfo.value.errors()[0]["type"]
        == "filtered_chain_map.e0_output_digits_exceeded"
    )


def test_page_zero_output_estimate_separates_large_map_coefficients() -> None:
    from fractions import Fraction

    rank = 5
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=0,
        basis_sizes=(rank,),
        differential_matrices=(),
    )
    basis = tuple(
        tuple(1 if row == column else 0 for column in range(rank))
        for row in range(rank)
    )
    filtration = (FiltrationLevel(subspaces=(FilteredSubspace(vectors=basis),)),)
    large = Fraction(10**4095)
    diagonal = tuple(
        tuple(
            large if row == column == 0 else (1 if row == column else 0)
            for column in range(rank)
        )
        for row in range(rank)
    )
    request = FilteredChainMapRequest(
        chain_map=ChainMapValue(
            source=complex_value,
            target=complex_value,
            map_matrices=(diagonal,),
        ),
        source_filtration=filtration,
        target_filtration=filtration,
    )

    result = filtered_chain_map_page_zero(
        request.chain_map, request.source_filtration, request.target_filtration
    )

    assert result.maps[0][0][0][0] == large


def test_page_zero_map_tool_has_the_native_result_contract() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "homological.filtered_chain_map.page_zero.compute"
    )
    assert isinstance(tool, MathTool)
    assert tool.result_type is FilteredChainMapPageZeroResult
