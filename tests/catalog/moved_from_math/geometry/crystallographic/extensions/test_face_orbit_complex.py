"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/geometry/crystallographic/extensions/test_face_orbit_complex.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry.crystallographic.extensions._models import (
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


def _compute(*, klein: bool):
    request = _polygon_request(klein=klein)
    pairing = pair_crystallographic_polytope_facets(
        request.affine_realization,
        request.polytope,
        request.lattice_axes,
        request.pairings,
    )
    domain = check_crystallographic_fundamental_domain(pairing)
    assert domain.is_fundamental_domain
    return quotient_face_orbit_complex(domain)


def test_face_orbit_operation_is_discoverable_and_recomputes_source() -> None:
    tool = next(
        item
        for item in BUILTIN_TOOLS
        if item.operation_id == "crystallographic.quotient_face_orbits.compute"
    )
    source = _compute(klein=False).source
    # ``dispatch.parse_operation_input`` is exactly this projection, inlined so a
    # math test does not import the product dispatch boundary.
    parsed = tool.request_type.model_validate_json(
        encode_strict_json(source.model_dump(mode="json")), strict=True
    )

    assert tool.run(parsed).quotient_chain_complex.basis_sizes == (1, 2, 1)
