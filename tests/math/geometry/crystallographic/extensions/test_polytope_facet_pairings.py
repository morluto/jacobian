"""Exact side pairing ledgers bound to crystallographic affine actions."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import parse_operation_input
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
from jacobian.math.geometry.polytopes._models import RationalVPolytope


def _request() -> CrystallographicPolytopePairingRequest:
    source = FiniteLatticeExtension(
        multiplication_table=((0,),),
        action_matrices=(((1, 0), (0, 1)),),
        factor_set=(((0, 0),),),
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
            for index, point in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
        ),
    )
    return CrystallographicPolytopePairingRequest(
        affine_realization=affine_section_realization(source),
        polytope=polytope,
        lattice_axes=("x", "y"),
        pairings=(
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
        ),
    )


def test_unit_square_translation_pairings_are_exact_and_source_bound() -> None:
    result = pair_crystallographic_polytope_facets(_request())

    assert result.polytope == _request().polytope
    assert result.affine_realization == _request().affine_realization
    assert result.lattice_axes == ("x", "y")
    assert tuple(
        (
            tuple(
                Fraction(*value.as_integer_ratio())
                for value in facet.halfspace.coefficients
            ),
            Fraction(*facet.halfspace.offset.as_integer_ratio()),
        )
        for facet in result.facet_profile.facets
    ) == (
        ((-1, 0), 0),
        ((0, -1), 0),
        ((0, 1), 1),
        ((1, 0), 1),
    )
    assert tuple(
        (item.source_facet_index, item.target_facet_index, item.lattice_translation)
        for item in result.pairings
    ) == (
        (0, 3, (1, 0)),
        (1, 2, (0, 1)),
        (2, 1, (0, -1)),
        (3, 0, (-1, 0)),
    )


def test_pairing_rejects_wrong_axis_binding() -> None:
    request = _request().model_copy(update={"lattice_axes": ("y", "x")})

    with pytest.raises(OperationDomainValidationError, match="ordered polytope axes"):
        pair_crystallographic_polytope_facets(request)


def test_pairing_rejects_noninverse_or_mismatched_facet_map() -> None:
    request = _request()
    pairings = list(request.pairings)
    pairings[3] = pairings[3].model_copy(update={"lattice_translation": (0, 0)})
    request = request.model_copy(update={"pairings": tuple(pairings)})

    with pytest.raises(
        OperationDomainValidationError, match="exact two-sided group inverse"
    ):
        pair_crystallographic_polytope_facets(request)


def test_pairing_rejects_inverse_elements_that_miss_target_facet() -> None:
    request = _request()
    pairings = list(request.pairings)
    pairings[0] = pairings[0].model_copy(update={"lattice_translation": (2, 0)})
    pairings[3] = pairings[3].model_copy(update={"lattice_translation": (-2, 0)})
    request = request.model_copy(update={"pairings": tuple(pairings)})

    with pytest.raises(
        OperationDomainValidationError,
        match="complete source facet vertex set",
    ):
        pair_crystallographic_polytope_facets(request)


def test_pairing_rejects_incomplete_ledger() -> None:
    request = _request().model_copy(update={"pairings": _request().pairings[:-1]})

    with pytest.raises(
        OperationDomainValidationError, match="one entry per computed facet"
    ):
        pair_crystallographic_polytope_facets(request)


def test_pairing_operation_is_published_with_square_example() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id
        == "crystallographic.extension.polytope_facet_pairings.compute"
    )

    assert tool.examples[0].name == "unit_square_translation_pairings"
    example_request = parse_operation_input(tool.request_type, tool.examples[0].input)
    assert tool.run(example_request).facet_profile.dimension == 2


def test_unit_square_is_fundamental_domain_including_boundary_only_contacts() -> None:
    result = check_crystallographic_fundamental_domain(
        pair_crystallographic_polytope_facets(_request())
    )
    assert result.is_fundamental_domain
    assert result.polytope_volume.as_fraction() == 1
    assert result.quotient_covolume.as_fraction() == 1
    assert result.overlap_translation is None


def test_width_two_square_fails_with_full_dimensional_translate_witness() -> None:
    request = _request()
    wide_vertices = tuple(
        vertex.model_copy(
            update={
                "coordinates": (
                    CanonicalRational.from_fraction(
                        vertex.coordinates[0].as_fraction() * 2
                    ),
                    vertex.coordinates[1],
                )
            }
        )
        for vertex in request.polytope.vertices
    )
    wide_polytope = request.polytope.model_copy(update={"vertices": wide_vertices})
    pairings = tuple(
        pairing.model_copy(
            update={
                "lattice_translation": (
                    pairing.lattice_translation[0] * 2,
                    pairing.lattice_translation[1],
                )
            }
        )
        for pairing in request.pairings
    )
    wide_request = request.model_copy(
        update={"polytope": wide_polytope, "pairings": pairings}
    )
    result = check_crystallographic_fundamental_domain(
        pair_crystallographic_polytope_facets(wide_request)
    )
    assert not result.is_fundamental_domain
    assert result.overlap_translation is not None
    assert result.overlap_holonomy_element == 0


def test_off_origin_rational_square_is_fundamental_domain() -> None:
    request = _request()
    vertices = tuple(
        vertex.model_copy(
            update={
                "coordinates": (
                    CanonicalRational.from_fraction(
                        vertex.coordinates[0].as_fraction() + Fraction(1, 2)
                    ),
                    CanonicalRational.from_fraction(
                        vertex.coordinates[1].as_fraction() - Fraction(1, 3)
                    ),
                )
            }
        )
        for vertex in request.polytope.vertices
    )
    moved = request.model_copy(
        update={"polytope": request.polytope.model_copy(update={"vertices": vertices})}
    )
    result = check_crystallographic_fundamental_domain(
        pair_crystallographic_polytope_facets(moved)
    )
    assert result.is_fundamental_domain
    assert result.polytope_volume.as_fraction() == 1


def test_fundamental_domain_check_rejects_stale_facet_profile() -> None:
    checked = pair_crystallographic_polytope_facets(_request())
    profile = checked.facet_profile
    first = profile.facets[0]
    stale_facet = first.model_copy(
        update={
            "halfspace": first.halfspace.model_copy(
                update={"offset": CanonicalRational.from_integer_ratio(1, 1)}
            )
        }
    )
    stale_profile = profile.model_copy(
        update={"facets": (stale_facet, *profile.facets[1:])}
    )
    stale = checked.model_copy(update={"facet_profile": stale_profile})
    with pytest.raises(OperationDomainValidationError):
        check_crystallographic_fundamental_domain(stale)


def test_fundamental_domain_check_rejects_stale_group_claim() -> None:
    checked = pair_crystallographic_polytope_facets(_request())
    source = checked.affine_realization.source.model_copy(
        update={"factor_set": (((1, 0),),)}
    )
    realization = checked.affine_realization.model_copy(update={"source": source})
    stale = checked.model_copy(update={"affine_realization": realization})
    with pytest.raises(OperationDomainValidationError):
        check_crystallographic_fundamental_domain(stale)
