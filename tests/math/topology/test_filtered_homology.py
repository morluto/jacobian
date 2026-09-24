from __future__ import annotations

from itertools import product

import pytest

from jacobian.catalog.models import MathTool
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredChainComplexRequest,
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    FilteredHomologyResult,
    filtered_homology_filtration,
)
from jacobian.math.topology.chain_complexes.filtered_extensions_tools import TOOLS
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)


def _request() -> FilteredChainComplexRequest:
    complex_value = ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=2,
        degree_min=0,
        degree_max=1,
        basis_sizes=(3, 1),
        differential_matrices=(((1,), (0,), (0,)),),
    )
    return FilteredChainComplexRequest(
        complex=complex_value,
        filtration=(
            FiltrationLevel(
                subspaces=(
                    FilteredSubspace(vectors=((1, 1, 1),)),
                    FilteredSubspace(vectors=()),
                )
            ),
            FiltrationLevel(
                subspaces=(
                    FilteredSubspace(vectors=((1, 0, 0), (0, 1, 0), (0, 0, 1))),
                    FilteredSubspace(vectors=((1,),)),
                )
            ),
        ),
    )


def _span(vectors: tuple[tuple[int, ...], ...]) -> set[tuple[int, ...]]:
    if not vectors:
        return {()}
    return {
        tuple(
            sum(coefficients[i] * vector[j] for i, vector in enumerate(vectors)) % 2
            for j in range(len(vectors[0]))
        )
        for coefficients in product((0, 1), repeat=len(vectors))
    }


def test_diagonal_filtered_image_is_not_a_coordinate_splitting() -> None:
    result = filtered_homology_filtration(_request())

    h0 = result.homology[0]
    assert h0.cycle_basis == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert h0.boundary_basis == ((1, 0, 0),)
    assert h0.homology_basis == ((0, 1, 0), (0, 0, 1))

    first_image = result.image_filtration[0].subspaces[0]
    assert first_image.basis_coordinates == ((1, 1),)
    assert first_image.cycle_representatives == ((1, 1, 1),)
    assert first_image.boundary_preimages == ((1,),)
    # The induced subspace is diagonal in the retained H_0 basis. The result
    # reports its exact inclusion and makes no direct-sum splitting claim.
    assert first_image.basis_coordinates[0] not in ((1, 0), (0, 1))
    assert result.image_filtration[1].subspaces[0].basis_coordinates == (
        (1, 0),
        (0, 1),
    )

    restored = FilteredHomologyResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_image_and_boundary_witnesses_match_exhaustive_gf2_oracle() -> None:
    result = filtered_homology_filtration(_request())
    returned_coordinates = tuple(
        tuple(int(value) for value in vector)
        for vector in result.image_filtration[0].subspaces[0].basis_coordinates
    )
    source_filtered_vectors = _span(((1, 1, 1),))
    oracle_image = {
        (vector[1] % 2, vector[2] % 2) for vector in source_filtered_vectors
    }
    returned_image = _span(returned_coordinates)
    assert returned_image == oracle_image == {(0, 0), (1, 1)}

    # Replay d(y) = representative - inclusion(homology coordinates).
    representative = tuple(
        int(v) for v in result.image_filtration[0].subspaces[0].cycle_representatives[0]
    )
    preimage = tuple(
        int(v) for v in result.image_filtration[0].subspaces[0].boundary_preimages[0]
    )
    image = (preimage[0], 0, 0)
    homology_coordinates = returned_coordinates[0]
    included = (
        homology_coordinates[0],
        homology_coordinates[1],
    )
    assert (
        tuple((a - b) % 2 for a, b in zip(representative, (0, *included), strict=True))
        == image
    )


def test_filtered_homology_tool_is_published_with_exact_result_type() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id
        == "homological.filtered_chain_complex.homology_filtration.compute"
    )
    assert isinstance(tool, MathTool)
    assert tool.result_type is FilteredHomologyResult
    assert tool.run(_request()).homology[0].homology_basis == (
        (0, 1, 0),
        (0, 0, 1),
    )


def test_filtered_homology_rejects_rational_domain_until_growth_is_bounded() -> None:
    request = _request().model_dump(mode="python")
    request["complex"]["coefficient_ring"] = CoefficientRing.RATIONAL
    request["complex"]["prime"] = None
    rational_request = FilteredChainComplexRequest.model_validate(request)
    with pytest.raises(ValueError, match=r"bounded GF\(p\)"):
        filtered_homology_filtration(rational_request)
