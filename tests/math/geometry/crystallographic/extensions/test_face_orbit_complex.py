"""Exact quotient-chain fixtures from verified Bieberbach polygons."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
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
from jacobian.math.geometry.polytopes._models import RationalVPolytope
from jacobian.math.topology.chain_complexes.operations import homology_groups


def _polygon_request(*, klein: bool) -> CrystallographicPolytopePairingRequest:
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
    return CrystallographicPolytopePairingRequest(
        affine_realization=affine_section_realization(extension),
        polytope=polytope,
        lattice_axes=("x", "y"),
        pairings=pairings,
    )


def _pair_polygon(*, klein: bool):
    request = _polygon_request(klein=klein)
    return pair_crystallographic_polytope_facets(
        request.affine_realization,
        request.polytope,
        request.lattice_axes,
        request.pairings,
    )


def _compute(*, klein: bool):
    pairing = _pair_polygon(klein=klein)
    domain = check_crystallographic_fundamental_domain(pairing)
    assert domain.is_fundamental_domain
    return quotient_face_orbit_complex(domain)


def test_unit_square_face_orbits_augment_to_torus_chains() -> None:
    result = _compute(klein=False)

    assert result.vertex_orbits == ((0, 1, 2, 3),)
    assert result.edge_orbit_representatives == (0, 1)
    assert len(result.orbit_maps) == 8
    assert result.quotient_chain_complex.basis_sizes == (1, 2, 1)
    assert result.quotient_chain_complex.differential_matrices == (
        ((0, 0),),
        ((0,), (0,)),
    )
    homology = homology_groups(result.quotient_chain_complex).homology_groups
    assert tuple(group.free_rank for group in homology) == (1, 2, 1)


def test_klein_bottle_face_orbits_give_integral_homology() -> None:
    result = _compute(klein=True)
    chain = result.quotient_chain_complex

    assert chain.basis_sizes == (1, 2, 1)
    assert chain.differential_matrices[0] == ((0, 0),)
    assert sorted(row[0] for row in chain.differential_matrices[1]) == [0, 2]
    homology = homology_groups(chain).homology_groups
    assert homology[0].free_rank == 1
    assert homology[1].free_rank == 1
    assert homology[1].torsion_invariant_factors == (2,)
    assert homology[2].free_rank == 0


def test_face_orbit_value_rejects_incomplete_endpoint_map_ledger() -> None:
    result = _compute(klein=False)
    forged = result.model_copy(update={"orbit_maps": result.orbit_maps[:-1]})

    with pytest.raises(ValidationError, match="complete endpoint maps"):
        BieberbachFaceOrbitComplex.model_validate(
            forged.model_dump(mode="python", warnings=False), strict=True
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("lattice_translation", (0,)), ("holonomy_element", 1)),
)
def test_face_orbit_value_rejects_labels_outside_retained_extension(
    field: str, value: object
) -> None:
    result = _compute(klein=False)
    entry = result.boundary_1_to_0[0].model_copy(update={field: value})
    forged = result.model_copy(
        update={"boundary_1_to_0": (entry, *result.boundary_1_to_0[1:])}
    )

    with pytest.raises(ValidationError):
        BieberbachFaceOrbitComplex.model_validate(
            forged.model_dump(mode="python", warnings=False), strict=True
        )


def test_face_orbit_operation_is_discoverable_and_recomputes_source() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "crystallographic.quotient_face_orbits.compute"
    )
    source = _compute(klein=False).source
    parsed = parse_operation_input(tool.request_type, source.model_dump(mode="json"))

    assert tool.run(parsed).quotient_chain_complex.basis_sizes == (1, 2, 1)
