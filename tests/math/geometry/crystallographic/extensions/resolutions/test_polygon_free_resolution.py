"""Exact free resolutions from the bounded Bieberbach polygon carrier."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.dispatch import parse_operation_input
from jacobian.math.geometry.crystallographic.extensions._models import (
    BieberbachFaceOrbitComplex,
    CrystallographicPolytopePairingRequest,
    FiniteLatticeExtension,
    PolytopeFacetPairing,
)
from jacobian.math.geometry.crystallographic.extensions.face_orbits import (
    quotient_face_orbit_complex,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
    check_crystallographic_fundamental_domain,
    pair_crystallographic_polytope_facets,
)
from jacobian.math.geometry.crystallographic.extensions.resolutions._models import (
    BieberbachPolygonFreeResolutionRequest,
)
from jacobian.math.geometry.crystallographic.extensions.resolutions.operations import (
    polygon_free_resolution,
)
from jacobian.math.geometry.polytopes._models import RationalVPolytope
from jacobian.math.topology.chain_complexes.operations import homology_groups


def _face_orbits(*, klein: bool) -> BieberbachFaceOrbitComplex:
    if klein:
        extension = FiniteLatticeExtension(
            multiplication_table=((0, 1), (1, 0)),
            action_matrices=(
                ((1, 0), (0, 1)),
                ((1, 0), (0, -1)),
            ),
            factor_set=(((0, 0), (0, 0)), ((0, 0), (1, 0))),
        )
        points = ((0, 0), (0, 1), (Fraction(1, 2), 0), (Fraction(1, 2), 1))
        pairings = (
            PolytopeFacetPairing(
                source_facet_index=0,
                target_facet_index=3,
                lattice_translation=(0, 1),
                holonomy_element=1,
            ),
            PolytopeFacetPairing(
                source_facet_index=1,
                target_facet_index=2,
                lattice_translation=(0, 1),
                holonomy_element=0,
            ),
            PolytopeFacetPairing(
                source_facet_index=2,
                target_facet_index=1,
                lattice_translation=(0, -1),
                holonomy_element=0,
            ),
            PolytopeFacetPairing(
                source_facet_index=3,
                target_facet_index=0,
                lattice_translation=(-1, 1),
                holonomy_element=1,
            ),
        )
    else:
        extension = FiniteLatticeExtension(
            multiplication_table=((0,),),
            action_matrices=(((1, 0), (0, 1)),),
            factor_set=(((0, 0),),),
        )
        points = ((0, 0), (0, 1), (1, 0), (1, 1))
        pairings = (
            PolytopeFacetPairing(
                source_facet_index=0,
                target_facet_index=3,
                lattice_translation=(1, 0),
                holonomy_element=0,
            ),
            PolytopeFacetPairing(
                source_facet_index=1,
                target_facet_index=2,
                lattice_translation=(0, 1),
                holonomy_element=0,
            ),
            PolytopeFacetPairing(
                source_facet_index=2,
                target_facet_index=1,
                lattice_translation=(0, -1),
                holonomy_element=0,
            ),
            PolytopeFacetPairing(
                source_facet_index=3,
                target_facet_index=0,
                lattice_translation=(-1, 0),
                holonomy_element=0,
            ),
        )
    polytope = RationalVPolytope(
        space={"axes": ("x", "y")},
        vertices=tuple(
            {
                "vertex_id": f"v{index}",
                "coordinates": tuple(
                    CanonicalRational.from_fraction(Fraction(value)) for value in point
                ),
            }
            for index, point in enumerate(points)
        ),
    )
    pairing = pair_crystallographic_polytope_facets(
        CrystallographicPolytopePairingRequest(
            affine_realization=affine_section_realization(extension),
            polytope=polytope,
            lattice_axes=("x", "y"),
            pairings=pairings,
        )
    )
    domain = check_crystallographic_fundamental_domain(pairing)
    assert domain.is_fundamental_domain
    return quotient_face_orbit_complex(domain)


@pytest.mark.parametrize("klein", (False, True))
def test_free_resolution_augments_to_the_known_quotient_chains(klein: bool) -> None:
    face_orbits = _face_orbits(klein=klein)
    resolution = polygon_free_resolution(face_orbits)

    assert resolution.group == face_orbits.source.source.affine_realization.source
    assert tuple(
        (entry.source_cell_index, entry.coefficient)
        for entry in resolution.augmentation
    ) == tuple((index, 1) for index in range(resolution.vertex_orbit_count))
    assert resolution.boundary_1_to_0 == face_orbits.boundary_1_to_0
    assert resolution.boundary_2_to_1 == face_orbits.boundary_2_to_1
    assert resolution.augmented_chain_complex == face_orbits.quotient_chain_complex
    restored = type(resolution).model_validate_json(resolution.model_dump_json())
    assert restored == resolution

    homology = homology_groups(resolution.augmented_chain_complex).homology_groups
    if klein:
        # The presentation <a,b | a b a^-1 = b^-1> independently gives
        # H_1 = abelianization = Z + Z/2.
        assert homology[1].free_rank == 1
        assert homology[1].torsion_invariant_factors == (2,)
    else:
        assert homology[1].free_rank == 2
        assert homology[1].torsion_invariant_factors == ()


def test_free_resolution_operation_is_discoverable_and_checks_the_source() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    source = _face_orbits(klein=False)
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id
        == "crystallographic.bieberbach.polygon_free_resolution.compute"
    )
    parsed = parse_operation_input(
        tool.request_type,
        BieberbachPolygonFreeResolutionRequest(source=source).model_dump(mode="json"),
    )
    assert tool.run(parsed) == polygon_free_resolution(source)

    forged = source.model_copy(update={"boundary_2_to_1": source.boundary_2_to_1[:-1]})
    with pytest.raises(ValidationError):
        BieberbachFaceOrbitComplex.model_validate(
            forged.model_dump(mode="python", warnings=False), strict=True
        )
