"""Exact product-cell oracle for checked translation tori."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry.crystallographic.extensions._models import (
    CrystallographicPolytopePairingRequest,
    FiniteLatticeExtension,
    PolytopeFacetPairing,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
    check_crystallographic_fundamental_domain,
    pair_crystallographic_polytope_facets,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori.operations import (
    translation_torus_quotient_chains,
)
from jacobian.math.geometry.polytopes._models import RationalVPolytope
from jacobian.math.geometry.polytopes.operations import facet_incidence
from jacobian.math.topology.chain_complexes.operations import homology_groups


def _checked_translation_torus(generators: tuple[tuple[int, ...], ...]):
    dimension = len(generators)
    extension = FiniteLatticeExtension(
        multiplication_table=((0,),),
        action_matrices=(
            tuple(
                tuple(int(row == column) for column in range(dimension))
                for row in range(dimension)
            ),
        ),
        factor_set=((tuple(0 for _ in range(dimension)),),),
    )
    points_by_mask = {
        tuple(
            sum(
                generators[axis][coordinate]
                for axis in range(dimension)
                if mask & (1 << axis)
            )
            for coordinate in range(dimension)
        ): mask
        for mask in range(1 << dimension)
    }
    points = tuple(points_by_mask)
    polytope = RationalVPolytope(
        space={"axes": tuple(f"x{axis}" for axis in range(dimension))},
        vertices=tuple(
            {
                "vertex_id": f"v{index:02d}",
                "coordinates": tuple(
                    CanonicalRational.from_fraction(value) for value in point
                ),
            }
            for index, point in enumerate(points)
        ),
    )
    profile = facet_incidence(polytope.vertices, dimension_bound=dimension)
    facets_by_bit: dict[tuple[int, int], int] = {}
    for index, facet in enumerate(profile.facets):
        masks = {
            points_by_mask[
                tuple(
                    value.as_fraction()
                    for value in profile.vertices[vertex].coordinates
                )
            ]
            for vertex in facet.source_vertex_indices
        }
        fixed_bits = [
            axis
            for axis in range(dimension)
            if all(mask & (1 << axis) for mask in masks)
            or all(not mask & (1 << axis) for mask in masks)
        ]
        assert len(fixed_bits) == 1
        fixed_bit = fixed_bits[0]
        fixed_value = int(all(mask & (1 << fixed_bit) for mask in masks))
        facets_by_bit[(fixed_bit, fixed_value)] = index
    pairings = []
    for axis in range(dimension):
        low = facets_by_bit[(axis, 0)]
        high = facets_by_bit[(axis, 1)]
        direction = generators[axis]
        pairings.extend(
            (
                PolytopeFacetPairing(
                    source_facet_index=low,
                    target_facet_index=high,
                    lattice_translation=direction,
                    holonomy_element=0,
                ),
                PolytopeFacetPairing(
                    source_facet_index=high,
                    target_facet_index=low,
                    lattice_translation=tuple(-value for value in direction),
                    holonomy_element=0,
                ),
            )
        )
    request = CrystallographicPolytopePairingRequest(
        affine_realization=affine_section_realization(extension),
        polytope=polytope,
        lattice_axes=tuple(f"x{axis}" for axis in range(dimension)),
        pairings=tuple(pairings),
    )
    checked = check_crystallographic_fundamental_domain(
        pair_crystallographic_polytope_facets(request)
    )
    assert checked.is_fundamental_domain
    return checked


def _checked_cube():
    return _checked_translation_torus(((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_translation_torus_operation_is_published() -> None:
    operation = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id
        == "crystallographic.translation_torus.quotient_chains.compute"
    )
    source = _checked_cube()
    result = operation.run(source)

    assert operation.request_type is type(source)
    assert operation.result_type is type(result)


def test_checked_cube_gives_integral_torus_product_chains() -> None:
    result = translation_torus_quotient_chains(_checked_cube())
    chain = result.quotient_chain_complex

    assert chain.basis_sizes == (1, 3, 3, 1)
    assert chain.differential_matrices == (
        (("0", "0", "0"),),
        (("0", "0", "0"), ("0", "0", "0"), ("0", "0", "0")),
        (("0",), ("0",), ("0",)),
    )
    assert tuple(
        group.free_rank for group in homology_groups(chain).homology_groups
    ) == (
        1,
        3,
        3,
        1,
    )


def test_serialized_source_composes_without_losing_exact_cell_data() -> None:
    source = _checked_cube()
    decoded = type(source).model_validate_json(source.model_dump_json())
    result = translation_torus_quotient_chains(decoded)

    assert tuple(
        tuple(value.as_fraction() for value in vector)
        for vector in result.circle_directions
    ) == ((0, 0, 1), (0, 1, 0), (1, 0, 0))
    assert result.source == decoded


def test_result_value_rejects_nonzero_differential() -> None:
    result = translation_torus_quotient_chains(_checked_cube())
    payload = result.model_dump(mode="python")
    chain = payload["quotient_chain_complex"]
    chain["differential_matrices"] = (
        (("1", "0", "0"),),
        *chain["differential_matrices"][1:],
    )

    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_checked_sheared_parallelepiped_recovers_integral_directions() -> None:
    generators = ((1, 1, 0), (0, 1, 1), (0, 0, 1))
    result = translation_torus_quotient_chains(_checked_translation_torus(generators))

    assert tuple(
        tuple(value.as_fraction() for value in vector)
        for vector in result.circle_directions
    ) == ((0, 0, 1), (0, 1, 1), (1, 1, 0))
    assert tuple(
        group.free_rank
        for group in homology_groups(result.quotient_chain_complex).homology_groups
    ) == (1, 3, 3, 1)


def test_four_dimensional_translation_torus_chains() -> None:
    result = translation_torus_quotient_chains(
        _checked_translation_torus(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        )
    )

    assert result.quotient_chain_complex.basis_sizes == (1, 4, 6, 4, 1)
    assert tuple(
        group.free_rank
        for group in homology_groups(result.quotient_chain_complex).homology_groups
    ) == (1, 4, 6, 4, 1)


@pytest.mark.parametrize(
    ("generators", "expected_ranks"),
    (
        (((1,),), (1, 1)),
        (((1, 0), (0, 1)), (1, 2, 1)),
    ),
)
def test_lower_rank_translation_tori(
    generators: tuple[tuple[int, ...], ...], expected_ranks: tuple[int, ...]
) -> None:
    result = translation_torus_quotient_chains(_checked_translation_torus(generators))

    assert (
        tuple(
            group.free_rank
            for group in homology_groups(result.quotient_chain_complex).homology_groups
        )
        == expected_ranks
    )
