"""Exact product-cell oracle for a checked translation 3-torus."""

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.crystallographic.extensions._models import (
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


def _checked_cube():
    extension = FiniteLatticeExtension(
        multiplication_table=((0,),),
        action_matrices=(((1, 0, 0), (0, 1, 0), (0, 0, 1)),),
        factor_set=(((0, 0, 0),),),
    )
    points = tuple(product((0, 1), repeat=3))
    polytope = RationalVPolytope(
        space={"axes": ("x", "y", "z")},
        vertices=tuple(
            {
                "vertex_id": f"v{index}",
                "coordinates": tuple(
                    CanonicalRational.from_fraction(value) for value in point
                ),
            }
            for index, point in enumerate(points)
        ),
    )
    profile = facet_incidence(polytope.vertices, dimension_bound=3)
    facets_by_side = {}
    for index, facet in enumerate(profile.facets):
        coordinates = [
            tuple(value.as_fraction() for value in profile.vertices[vertex].coordinates)
            for vertex in facet.source_vertex_indices
        ]
        fixed_axes = [
            axis
            for axis in range(3)
            if len({point[axis] for point in coordinates}) == 1
        ]
        assert len(fixed_axes) == 1
        axis = fixed_axes[0]
        facets_by_side[(axis, coordinates[0][axis])] = index
    pairings = []
    for axis in range(3):
        low = facets_by_side[(axis, 0)]
        high = facets_by_side[(axis, 1)]
        unit = tuple(int(coordinate == axis) for coordinate in range(3))
        pairings.extend(
            (
                PolytopeFacetPairing(
                    source_facet_index=low,
                    target_facet_index=high,
                    lattice_translation=unit,
                    holonomy_element=0,
                ),
                PolytopeFacetPairing(
                    source_facet_index=high,
                    target_facet_index=low,
                    lattice_translation=tuple(-value for value in unit),
                    holonomy_element=0,
                ),
            )
        )
    pairing = pair_crystallographic_polytope_facets(
        affine_section_realization(extension),
        polytope,
        ("x", "y", "z"),
        tuple(pairings),
    )
    checked = check_crystallographic_fundamental_domain(pairing)
    assert checked.is_fundamental_domain
    return checked


def test_checked_cube_gives_integral_torus_product_chains() -> None:
    result = translation_torus_quotient_chains(_checked_cube())
    chain = result.quotient_chain_complex

    assert chain.basis_sizes == (1, 3, 3, 1)
    assert chain.differential_matrices == (
        ((0, 0, 0),),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        ((0,), (0,), (0,)),
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
        ((1, 0, 0),),
        *chain["differential_matrices"][1:],
    )

    with pytest.raises(ValidationError):
        type(result).model_validate(payload)
