"""Exact comparison of a stable spectral page with graded homology."""

from __future__ import annotations

import json

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    SpectralAbutmentRequest,
    SpectralAbutmentResult,
    abutment,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)


def _filtration() -> tuple[FiltrationLevel, ...]:
    return (
        FiltrationLevel(
            subspaces=(FilteredSubspace(vectors=((1, 1),)),),
        ),
        FiltrationLevel(
            subspaces=(FilteredSubspace(vectors=((1, 0), (0, 1))),),
        ),
    )


@pytest.mark.parametrize(
    ("ring", "prime"),
    ((CoefficientRing.RATIONAL, None), (CoefficientRing.PRIME_FIELD, 5)),
)
def test_stable_page_maps_isomorphically_to_graded_homology(ring, prime) -> None:
    """A diagonal filtration yields one nontrivial graded class at each level."""
    complex_value = ChainComplexValue(
        coefficient_ring=ring,
        prime=prime,
        degree_min=0,
        degree_max=0,
        basis_sizes=(2,),
        differential_matrices=(),
    )
    request = SpectralAbutmentRequest(
        complex=complex_value,
        filtration=_filtration(),
    )

    result = abutment(request)

    assert result.status == "STABILIZED"
    assert result.page.page == 1
    assert [
        comparison.page_dimension
        for level in result.comparisons
        for comparison in level
    ] == [1, 1]
    assert [
        comparison.homology_graded_dimension
        for level in result.comparisons
        for comparison in level
    ] == [1, 1]
    assert [
        comparison.matrix for level in result.comparisons for comparison in level
    ] == [((1,),), ((1,),)]
    assert result.comparisons[0][0].homology_graded_basis_coordinates == ((1, 1),)
    # This graded class is the diagonal line inside H_0, with no chosen
    # direct-sum splitting of the filtered homology.
    assert result.homology[0].homology_basis == ((1, 0), (0, 1))
    assert "splitting" not in result.model_dump_json()
    restored = SpectralAbutmentResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_abutment_tool_example_serializes_as_a_typed_result() -> None:
    from jacobian.math.topology.chain_complexes.filtered_extensions_tools import TOOLS

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "homological.spectral_sequence.abutment.compute"
    )
    request = SpectralAbutmentRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    result = tool.run(request)
    assert isinstance(result, SpectralAbutmentResult)
    assert result.page.page == 0
    assert all(
        comparison.page_dimension == comparison.homology_graded_dimension == 1
        and comparison.matrix == ((1,),)
        for level in result.comparisons
        for comparison in level
    )


def test_acyclic_complex_has_empty_graded_homology_comparison() -> None:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        prime=None,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 1),
        differential_matrices=(((1,),),),
    )
    filtration = (
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

    result = abutment(
        SpectralAbutmentRequest(complex=complex_value, filtration=filtration)
    )

    assert result.page.page == 2
    assert all(
        comparison.page_dimension == comparison.homology_graded_dimension == 0
        and comparison.matrix == ()
        for level in result.comparisons
        for comparison in level
    )


def test_stable_page_representative_is_corrected_by_lower_filtration() -> None:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=5,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 2),
        differential_matrices=(((1, 1),),),
    )
    filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=((0, 1),)),
            ),
        ),
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=((1, 0), (0, 1))),
            ),
        ),
    )

    result = abutment(
        SpectralAbutmentRequest(complex=complex_value, filtration=filtration)
    )

    assert result.status == "STABILIZED"
    assert result.comparisons[1][1].page_dimension == 1
    assert result.comparisons[1][1].homology_graded_dimension == 1


def test_stable_page_representative_lift_preserves_rational_arithmetic() -> None:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        prime=None,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 2),
        differential_matrices=(((1, 1),),),
    )
    filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=((0, 1),)),
            ),
        ),
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=((1,),)),
                FilteredSubspace(vectors=((1, 0), (0, 1))),
            ),
        ),
    )

    result = abutment(
        SpectralAbutmentRequest(complex=complex_value, filtration=filtration)
    )

    assert result.status == "STABILIZED"
    assert result.comparisons[1][1].page_dimension == 1
    assert result.comparisons[1][1].homology_graded_dimension == 1


def test_abutment_preflights_complete_retained_output_cells() -> None:
    sizes = (10,) * 59
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=2,
        degree_min=-26,
        degree_max=32,
        basis_sizes=sizes,
        differential_matrices=tuple(
            tuple(tuple(0 for _ in range(10)) for _ in range(10))
            for _ in range(58)
        ),
    )
    spanning_vectors = tuple(
        tuple(int(row == column) for column in range(10))
        for row in range(10)
    )
    vectors = spanning_vectors + (spanning_vectors[0],) * 54
    filtration = tuple(
        FiltrationLevel(
            subspaces=tuple(
                FilteredSubspace(vectors=vectors) for _ in range(59)
            ),
        )
        for _ in range(8)
    )

    with pytest.raises(
        OperationResourceAdmissionError,
        match="complete abutment result retains up to",
    ) as error:
        abutment(
            SpectralAbutmentRequest(complex=complex_value, filtration=filtration)
        )
    assert error.value.errors()[0]["type"] == "spectral_sequence.abutment_output_bound"
