"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/geometry/crystallographic/extensions/translation_tori/test_translation_torus_chains.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

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
from jacobian.math.geometry.polytopes._models import RationalVPolytope
from jacobian.math.geometry.polytopes.operations import facet_incidence


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
    pairing = pair_crystallographic_polytope_facets(
        request.affine_realization,
        request.polytope,
        request.lattice_axes,
        request.pairings,
    )
    checked = check_crystallographic_fundamental_domain(pairing)
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
